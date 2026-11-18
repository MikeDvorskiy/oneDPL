// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_UNSEQ_BACKEND_SIMD_SORT_H
#define _ONEDPL_UNSEQ_BACKEND_SIMD_SORT_H

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#if __has_include(<ranges>)
#    include <ranges>
#endif

#include "onedpl_config.h"
#include "utils.h"

// Size of a SIMD block in bytes. 64 bytes is one AVX-512 register; on narrower targets the block spans
// several registers, which was measured to be faster than a single narrow register (more independent work)
#ifndef _ONEDPL_SIMD_SORT_BLOCK_BYTES
#    define _ONEDPL_SIMD_SORT_BLOCK_BYTES 64
#endif

// SIMD sorting bricks: sorting networks and bitonic merges built from branchless compare-exchange
// operations on arrays, which the compiler maps to vector min/max (compare + blend) instructions.
//
// The bricks are applicable to integral keys (except bool) ordered by "less" or "greater" without
// a projection, stored in contiguous memory. Equal integral keys are indistinguishable, so the result
// is also valid for a stable sort.

namespace oneapi
{
namespace dpl
{
namespace __unseq_backend
{

//------------------------------------------------------------------------
// Applicability
//------------------------------------------------------------------------

// Ascending / descending order of the keys
template <bool __ascending>
struct __simd_sort_order
{
    static constexpr bool __is_ascending = __ascending;

    template <typename _Tp>
    bool
    operator()(const _Tp& __a, const _Tp& __b) const
    {
        if constexpr (__ascending)
            return __a < __b;
        else
            return __b < __a;
    }

    // The value which is ordered after (or equal to) any other value; used for padding
    template <typename _Tp>
    static constexpr _Tp
    __last_value()
    {
        return __ascending ? std::numeric_limits<_Tp>::max() : std::numeric_limits<_Tp>::lowest();
    }
};

// Maps a comparator to __simd_sort_order, or void if the comparator is not supported
template <typename _Compare, typename _Tp>
struct __simd_sort_order_of
{
    using type = void;
};

template <typename _Tp>
struct __simd_sort_order_of<std::less<_Tp>, _Tp>
{
    using type = __simd_sort_order<true>;
};
template <typename _Tp>
struct __simd_sort_order_of<std::less<>, _Tp>
{
    using type = __simd_sort_order<true>;
};
template <typename _Tp>
struct __simd_sort_order_of<std::greater<_Tp>, _Tp>
{
    using type = __simd_sort_order<false>;
};
template <typename _Tp>
struct __simd_sort_order_of<std::greater<>, _Tp>
{
    using type = __simd_sort_order<false>;
};
#if _ONEDPL_CPP20_RANGES_PRESENT
template <typename _Tp>
struct __simd_sort_order_of<std::ranges::less, _Tp>
{
    using type = __simd_sort_order<true>;
};
template <typename _Tp>
struct __simd_sort_order_of<std::ranges::greater, _Tp>
{
    using type = __simd_sort_order<false>;
};
// A comparator with identity projections, as created by the range-based algorithms
template <typename _Compare, typename _Tp>
struct __simd_sort_order_of<oneapi::dpl::__internal::__binary_op<_Compare, std::identity, std::identity>, _Tp>
    : __simd_sort_order_of<_Compare, _Tp>
{
};
#endif

template <typename _Iterator>
constexpr bool
__is_simd_sort_contiguous()
{
#if _ONEDPL_CPP20_RANGES_PRESENT
    return std::contiguous_iterator<_Iterator>;
#else
    return std::is_pointer_v<_Iterator>;
#endif
}

template <typename _Iterator, typename _Compare>
constexpr bool
__is_simd_sort_applicable()
{
    using _Tp = typename std::iterator_traits<_Iterator>::value_type;
    if constexpr (!std::is_integral_v<_Tp> || std::is_same_v<_Tp, bool> || !__is_simd_sort_contiguous<_Iterator>())
        return false;
    else
        return !std::is_void_v<typename __simd_sort_order_of<_Compare, _Tp>::type>;
}

template <typename _Iterator>
auto*
__simd_sort_to_address(_Iterator __it)
{
#if _ONEDPL_CPP20_RANGES_PRESENT
    return std::to_address(__it);
#else
    return __it;
#endif
}

// Number of keys in a SIMD block: from 4 to 16 keys
template <typename _Tp>
inline constexpr std::size_t __simd_sort_width =
    std::clamp<std::size_t>(std::size_t(_ONEDPL_SIMD_SORT_BLOCK_BYTES) / sizeof(_Tp), 4, 16);

//------------------------------------------------------------------------
// Primitives
//------------------------------------------------------------------------

// Branchless compare-exchange of __a[i] and __b[i]: afterwards __a[i] precedes __b[i].
// A select-based pair (not min/max) keeps the operation a permutation.
template <typename _Tp, typename _Order>
inline void
__simd_compare_exchange(_Tp* __a, _Tp* __b, std::size_t __n, _Order __order)
{
    _ONEDPL_PRAGMA_SIMD
    for (std::size_t __i = 0; __i < __n; ++__i)
    {
        const _Tp __x = __a[__i];
        const _Tp __y = __b[__i];
        const bool __swap = __order(__y, __x);
        __a[__i] = __swap ? __y : __x;
        __b[__i] = __swap ? __x : __y;
    }
}

// One half-cleaner stage with the distance _Dist, followed by the stages with smaller distances.
// The distance is a template parameter so that the pairing of the keys is a compile-time permutation.
template <std::size_t _Width, std::size_t _Dist, typename _Tp, typename _Order>
inline void
__simd_bitonic_clean_stage(_Tp* __v, _Order __order)
{
    _Tp __lo[_Width / 2];
    _Tp __hi[_Width / 2];
    _ONEDPL_PRAGMA_SIMD
    for (std::size_t __j = 0; __j < _Width / 2; ++__j)
    {
        const std::size_t __i = (__j / _Dist) * 2 * _Dist + (__j % _Dist);
        const _Tp __x = __v[__i];
        const _Tp __y = __v[__i + _Dist];
        const bool __swap = __order(__y, __x);
        __lo[__j] = __swap ? __y : __x;
        __hi[__j] = __swap ? __x : __y;
    }
    for (std::size_t __j = 0; __j < _Width / 2; ++__j)
    {
        const std::size_t __i = (__j / _Dist) * 2 * _Dist + (__j % _Dist);
        __v[__i] = __lo[__j];
        __v[__i + _Dist] = __hi[__j];
    }
    if constexpr (_Dist > 1)
        __simd_bitonic_clean_stage<_Width, _Dist / 2>(__v, __order);
}

// Sorts a bitonic sequence of _Width keys (half-cleaner cascade)
template <std::size_t _Width, typename _Tp, typename _Order>
inline void
__simd_bitonic_clean(_Tp* __v, _Order __order)
{
    __simd_bitonic_clean_stage<_Width, _Width / 2>(__v, __order);
}

// Merges two sorted blocks of _Width keys: afterwards __x holds the first _Width keys and __y the rest,
// both sorted
template <std::size_t _Width, typename _Tp, typename _Order>
inline void
__simd_bitonic_merge(_Tp* __x, _Tp* __y, _Order __order)
{
    _ONEDPL_PRAGMA_SIMD
    for (std::size_t __i = 0; __i < _Width; ++__i)
    {
        const _Tp __a = __x[__i];
        const _Tp __b = __y[_Width - 1 - __i];
        const bool __swap = __order(__b, __a);
        __x[__i] = __swap ? __b : __a;
        __y[_Width - 1 - __i] = __swap ? __a : __b;
    }
    __simd_bitonic_clean<_Width>(__x, __order);
    __simd_bitonic_clean<_Width>(__y, __order);
}

//------------------------------------------------------------------------
// Block of __simd_sort_width keys held in a vector register
//------------------------------------------------------------------------

#if defined(__has_builtin)
#    if __has_builtin(__builtin_shufflevector) && (defined(__clang__) || defined(__GNUC__))
#        define _ONEDPL_SIMD_SORT_VECTOR_EXT 1
#    endif
#endif
#ifndef _ONEDPL_SIMD_SORT_VECTOR_EXT
#    define _ONEDPL_SIMD_SORT_VECTOR_EXT 0
#endif

#if _ONEDPL_SIMD_SORT_VECTOR_EXT
// GCC/Clang vector extensions: each stage of the bitonic merge is one shuffle, min, max and blend
// on registers. Plain min/max are a permutation for integral keys: equal keys are identical.
template <typename _Tp, typename _Order>
struct __simd_sort_block
{
    static constexpr std::size_t __w = __simd_sort_width<_Tp>;
    typedef _Tp __type __attribute__((vector_size(__w * sizeof(_Tp))));
    using __lanes = std::make_index_sequence<__w>;

    static __type
    __load(const _Tp* __p)
    {
        __type __v;
        std::memcpy(&__v, __p, sizeof(__type));
        return __v;
    }
    static void
    __store(_Tp* __p, const __type& __v)
    {
        std::memcpy(__p, &__v, sizeof(__type));
    }

    // Lane-wise: the key ordered first / second of __a and __b
    static __type
    __first(__type __a, __type __b)
    {
        const __type __m = (__type)(_Order::__is_ascending ? (__a < __b) : (__b < __a));
        return (__a & __m) | (__b & ~__m);
    }
    static __type
    __second(__type __a, __type __b)
    {
        const __type __m = (__type)(_Order::__is_ascending ? (__a < __b) : (__b < __a));
        return (__b & __m) | (__a & ~__m);
    }

    template <std::size_t... _Is>
    static __type
    __reverse(__type __v, std::index_sequence<_Is...>)
    {
        return __builtin_shufflevector(__v, __v, (__w - 1 - _Is)...);
    }
    template <std::size_t _Dist, std::size_t... _Is>
    static __type
    __swap_pairs(__type __v, std::index_sequence<_Is...>)
    {
        return __builtin_shufflevector(__v, __v, (_Is ^ _Dist)...);
    }
    // Lane i takes __lo[i] if it is the first lane of its pair, else __hi[i]
    template <std::size_t _Dist, std::size_t... _Is>
    static __type
    __blend(__type __lo, __type __hi, std::index_sequence<_Is...>)
    {
        return __builtin_shufflevector(__lo, __hi, ((_Is & _Dist) ? _Is + __w : _Is)...);
    }

    template <std::size_t _Dist>
    static __type
    __clean(__type __v)
    {
        const __type __p = __swap_pairs<_Dist>(__v, __lanes{});
        __v = __blend<_Dist>(__first(__v, __p), __second(__v, __p), __lanes{});
        if constexpr (_Dist > 1)
            return __clean<_Dist / 2>(__v);
        else
            return __v;
    }

    // Afterwards __x holds the first __w keys of __x and __y, and __y the rest, both sorted
    static void
    __merge(__type& __x, __type& __y)
    {
        const __type __yr = __reverse(__y, __lanes{});
        const __type __lo = __first(__x, __yr);
        const __type __hi = __second(__x, __yr);
        __x = __clean<__w / 2>(__lo);
        __y = __clean<__w / 2>(__hi);
    }
};
#else
// Portable fallback: arrays and loops
template <typename _Tp, typename _Order>
struct __simd_sort_block
{
    static constexpr std::size_t __w = __simd_sort_width<_Tp>;
    struct __type
    {
        _Tp __v[__w];
    };

    static __type
    __load(const _Tp* __p)
    {
        __type __b;
        std::memcpy(__b.__v, __p, sizeof(__b.__v));
        return __b;
    }
    static void
    __store(_Tp* __p, const __type& __b)
    {
        std::memcpy(__p, __b.__v, sizeof(__b.__v));
    }
    static void
    __merge(__type& __x, __type& __y)
    {
        __simd_bitonic_merge<__w>(__x.__v, __y.__v, _Order{});
    }
};
#endif

// Branchless scalar merge, used for short sequences
template <typename _Tp, typename _Order>
inline void
__simd_scalar_merge(const _Tp* __a, std::size_t __na, const _Tp* __b, std::size_t __nb, _Tp* __result, _Order __order)
{
    std::size_t __ia = 0, __ib = 0;
    while (__ia < __na && __ib < __nb)
    {
        const bool __take_b = __order(__b[__ib], __a[__ia]);
        *__result++ = __take_b ? __b[__ib] : __a[__ia];
        __ib += __take_b;
        __ia += !__take_b;
    }
    if (__ia < __na)
        std::memcpy(__result, __a + __ia, (__na - __ia) * sizeof(_Tp));
    if (__ib < __nb)
        std::memcpy(__result, __b + __ib, (__nb - __ib) * sizeof(_Tp));
}

//------------------------------------------------------------------------
// Merge brick: [__a, __a + __na) and [__b, __b + __nb) into __result (must not overlap the inputs)
//------------------------------------------------------------------------

template <typename _Tp, typename _Order>
void
__simd_merge(const _Tp* __a, std::size_t __na, const _Tp* __b, std::size_t __nb, _Tp* __result, _Order __order)
{
    using _Block = __simd_sort_block<_Tp, _Order>;
    using _BlockType = typename _Block::__type;
    constexpr std::size_t __w = __simd_sort_width<_Tp>;
    if (__na < __w || __nb < __w)
    {
        __simd_scalar_merge(__a, __na, __b, __nb, __result, __order);
        return;
    }

    // The last partial block of each input is padded with the value ordered last. Integral keys
    // equal to it are indistinguishable from the padding, and the output is cut at __na + __nb keys.
    constexpr _Tp __pad = _Order::template __last_value<_Tp>();
    auto __load = [__pad](_BlockType& __dst, const _Tp* __src, std::size_t __rem) {
        if (__rem >= __w)
        {
            __dst = _Block::__load(__src);
            return __w;
        }
        _Tp __tmp[__w];
        std::memcpy(__tmp, __src, __rem * sizeof(_Tp));
        std::fill(__tmp + __rem, __tmp + __w, __pad);
        __dst = _Block::__load(__tmp);
        return __rem;
    };

    _BlockType __x = _Block::__load(__a);
    _BlockType __y = _Block::__load(__b);
    std::size_t __ia = __w;
    std::size_t __ib = __w;
    const std::size_t __total = __na + __nb;
    std::size_t __written = 0;

    // Main loop, while both inputs have a full block left: the blocks stay in registers, the stores have
    // a fixed size, and the next block (from the input with the smaller next key) is chosen without a branch
    while (__ia + __w <= __na && __ib + __w <= __nb)
    {
        _Block::__merge(__x, __y);
        _Block::__store(__result + __written, __x);
        __written += __w;

        const bool __take_a = !__order(__b[__ib], __a[__ia]);
        __x = _Block::__load(__take_a ? __a + __ia : __b + __ib);
        __ia += __take_a ? __w : 0;
        __ib += __take_a ? 0 : __w;
    }

    // Tail: at least one input has less than a full block left
    for (;;)
    {
        _Block::__merge(__x, __y);

        const std::size_t __k = std::min(__w, __total - __written);
        if (__k == __w)
            _Block::__store(__result + __written, __x);
        else
        {
            _Tp __tmp[__w];
            _Block::__store(__tmp, __x);
            std::memcpy(__result + __written, __tmp, __k * sizeof(_Tp));
        }
        __written += __k;
        if (__written == __total)
            break;

        const bool __has_a = __ia < __na;
        const bool __has_b = __ib < __nb;
        if (__has_a && (!__has_b || !__order(__b[__ib], __a[__ia])))
            __ia += __load(__x, __a + __ia, __na - __ia);
        else if (__has_b)
            __ib += __load(__x, __b + __ib, __nb - __ib);
        else
        {
            // Both inputs are consumed: flush the keys left in __y
            _Tp __tmp[__w];
            std::fill(__tmp, __tmp + __w, __pad);
            __x = _Block::__load(__tmp);
        }
    }
}

//------------------------------------------------------------------------
// Sort brick
//------------------------------------------------------------------------

// Sorts __w * __w keys at __t; __tmp must provide __w * __w keys of scratch space
template <typename _Tp, typename _Order>
void
__simd_sort_tile(_Tp* __t, _Tp* __tmp, _Order __order)
{
    constexpr std::size_t __w = __simd_sort_width<_Tp>;

    // 1. Bitonic sorting network applied to whole rows: each comparator is a vector compare-exchange of
    //    two rows, so all the __w columns are sorted at once
    for (std::size_t __k = 2; __k <= __w; __k *= 2)
    {
        for (std::size_t __base = 0; __base < __w; __base += __k)
            for (std::size_t __r = 0; __r < __k / 2; ++__r)
                __simd_compare_exchange(__t + (__base + __r) * __w, __t + (__base + __k - 1 - __r) * __w, __w, __order);
        for (std::size_t __d = __k / 4; __d > 0; __d /= 2)
            for (std::size_t __base = 0; __base < __w; __base += 2 * __d)
                for (std::size_t __r = 0; __r < __d; ++__r)
                    __simd_compare_exchange(__t + (__base + __r) * __w, __t + (__base + __r + __d) * __w, __w, __order);
    }

    // 2. Transpose: the sorted columns become sorted runs of __w keys
    for (std::size_t __r = 0; __r < __w; ++__r)
        _ONEDPL_PRAGMA_SIMD
    for (std::size_t __c = 0; __c < __w; ++__c)
        __tmp[__c * __w + __r] = __t[__r * __w + __c];

    // 3. Merge the runs
    _Tp* __src = __tmp;
    _Tp* __dst = __t;
    for (std::size_t __len = __w; __len < __w * __w; __len *= 2)
    {
        for (std::size_t __i = 0; __i < __w * __w; __i += 2 * __len)
            __simd_merge(__src + __i, __len, __src + __i + __len, __len, __dst + __i, __order);
        std::swap(__src, __dst);
    }
    if (__src != __t)
        std::memcpy(__t, __src, __w * __w * sizeof(_Tp));
}

// Number of keys sorted as a single tile
template <typename _Tp>
inline constexpr std::size_t __simd_sort_tile_size = __simd_sort_width<_Tp> * __simd_sort_width<_Tp>;

// Sorts [__first, __first + __n); __buf must provide __n keys of scratch space
template <typename _Tp, typename _Order>
void
__simd_sort(_Tp* __first, std::size_t __n, _Tp* __buf, _Order __order)
{
    constexpr std::size_t __tile = __simd_sort_tile_size<_Tp>;
    if (__n < __tile)
    {
        std::sort(__first, __first + __n, __order);
        return;
    }

    const std::size_t __full = __n / __tile * __tile;
    for (std::size_t __i = 0; __i < __full; __i += __tile)
        __simd_sort_tile(__first + __i, __buf + __i, __order);
    std::sort(__first + __full, __first + __n, __order);

    // Bottom-up merge of the sorted tiles
    _Tp* __src = __first;
    _Tp* __dst = __buf;
    for (std::size_t __len = __tile; __len < __n; __len *= 2)
    {
        for (std::size_t __i = 0; __i < __n; __i += 2 * __len)
        {
            const std::size_t __mid = std::min(__i + __len, __n);
            const std::size_t __end = std::min(__i + 2 * __len, __n);
            __simd_merge(__src + __i, __mid - __i, __src + __mid, __end - __mid, __dst + __i, __order);
        }
        std::swap(__src, __dst);
    }
    if (__src != __first)
        std::memcpy(__first, __src, __n * sizeof(_Tp));
}

// Sorts [__first, __first + __n) with a temporary buffer; falls back to std::sort if it cannot be allocated
template <typename _Tp, typename _Order>
void
__simd_sort(_Tp* __first, std::size_t __n, _Order __order) noexcept
{
    if (__n < __simd_sort_tile_size<_Tp>)
    {
        std::sort(__first, __first + __n, __order);
        return;
    }
    std::unique_ptr<_Tp[]> __buf(new (std::nothrow) _Tp[__n]);
    if (__buf)
        __simd_sort(__first, __n, __buf.get(), __order);
    else
        std::sort(__first, __first + __n, __order);
}

} // namespace __unseq_backend
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_UNSEQ_BACKEND_SIMD_SORT_H
