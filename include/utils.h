#ifndef __UTILS_H__
#define __UTILS_H__

#include <unordered_set>
#include <unordered_map>
#include <string>

namespace GeoRd::details {

template <typename T> double cos(const T &v1, const T &v2) {
    return v1.dot(v2) / (v1.norm() * v2.norm() + 1e-30);
}

template <typename T>
auto fp2str(T x) -> typename std::enable_if<std::is_floating_point<T>::value, std::string>::type {
    const double scale = 1.0e6;
    double x_scale = (x + 1e2) * scale;
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(8) << x_scale;
    return ss.str();
}

template <typename T>
auto hash_triplet_impl(T x, T y, T z) -> typename std::enable_if<std::is_floating_point<T>::value, std::string>::type {
    return fp2str<T>(x) + "," + fp2str<T>(y) + "," + fp2str<T>(z);
};

template <typename T>
auto hash_triplet_impl(T x, T y, T z) -> typename std::enable_if<std::is_integral<T>::value, std::string>::type {

    // sort
    int r1, r2, r3;
    r1 = std::min(std::min(x, y), z);
    r3 = std::max(std::max(x, y), z);
    r2 = x + y + z - r1 - r3;

    return std::to_string(r1) + "," + std::to_string(r2) + "," + std::to_string(r3);
};

template<typename>
std::false_type is_subscriptable_impl(...);
template<typename T>
auto is_subscriptable_impl(int) -> decltype(std::declval<T>()[0], std::true_type());

template <typename T> struct Subscriptable {
    static constexpr bool value = decltype(is_subscriptable_impl<T>(0))::value;
};


template <typename T> using Triplet = std::array<T, 3>;

template <typename T, typename = void> struct HashTable {
    std::size_t operator()(const T &t) const { return std::hash<T>()(t); }
};

template <typename T> struct HashTable<T, typename std::enable_if<Subscriptable<T>::value, void>::type> {
    std::size_t operator()(const T &p) const {
        std::hash<std::string> hash;
        return hash(hash_triplet_impl(p[0], p[1], p[2]));
    }
};


template <typename T>
auto key_equal_impl(const T &t1, const T &t2) -> typename std::enable_if<std::is_floating_point<T>::value, bool>::type {
    return std::abs(t1 - t2) < 1e-6;
}

template <typename T>
auto key_equal_impl(const T &t1, const T &t2) -> typename std::enable_if<std::is_integral<T>::value, bool>::type {
    return t1 == t2;
}

template <typename T, typename = void> struct KeyEqual {
    bool operator()(const T &t1, const T &t2) const { return t1 == t2; }
};

template <typename T> struct KeyEqual<T, typename std::enable_if<Subscriptable<T>::value>::type> {
    bool operator()(const T &p1, const T &p2) const {
        return key_equal_impl(p1[0], p2[0]) && key_equal_impl(p1[1], p2[1]) && key_equal_impl(p1[2], p2[2]);
    }
};

template <typename Key, typename Value>
using UnorderedMap =
    std::unordered_map<Key, Value, HashTable<Key>, KeyEqual<Key>>;

template <typename Key>
using UnorderedSet = std::unordered_set<Key, HashTable<Key>, KeyEqual<Key>>;

} // namespace GeoRd::details


#endif // __UTILS_H__