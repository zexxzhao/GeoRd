#ifndef __UTILS_H__
#define __UTILS_H__

#include <unordered_set>
#include <unordered_map>
#include <string>

namespace GeoRd::details {

template <typename T> double cos(const T &v1, const T &v2) {
    return v1.dot(v2) / (v1.norm() * v2.norm() + 1e-30);
}

inline std::string d2str(double x) {
    const double scale = 1.0e6;
    double x_scale = (x + 1e2) * scale;
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(8) << x_scale;
    return ss.str();
}

inline void hash_cor(double x, double y, double z, std::string &res_str) {
    std::string x_str, y_str, z_str;

    res_str.clear();
    x_str = d2str(x);
    y_str = d2str(y);
    z_str = d2str(z);

    res_str = x_str + "," + y_str + "," + z_str;
};

inline void hash_con(int x, int y, int z, std::string &res_str) {
    std::string x_str, y_str, z_str;

    int r1, r2, r3;
    r1 = std::min(std::min(x, y), z);
    r3 = std::max(std::max(x, y), z);
    r2 = x + y + z - r1 - r3;

    res_str.clear();
    x_str = std::to_string(r1);
    y_str = std::to_string(r2);
    z_str = std::to_string(r3);

    res_str = x_str + "," + y_str + "," + z_str;
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

template <typename T>
struct HashTable<Triplet<T>, typename std::enable_if<
                                 std::is_floating_point<T>::value>::type> {
    std::size_t operator()(const Triplet<T> &t) const {
        std::string str;
        hash_cor(std::get<0>(t), std::get<1>(t), std::get<2>(t), str);
        return std::hash<std::string>()(str);
    }
};

template <typename T>
struct HashTable<Triplet<T>,
                 typename std::enable_if<std::is_integral<T>::value>::type> {
    std::size_t operator()(const Triplet<T> &t) const {
        std::string str;
        hash_con(std::get<0>(t), std::get<1>(t), std::get<2>(t), str);
        return std::hash<std::string>()(str);
    }
};

template <typename T> struct HashTable<T, typename std::enable_if<Subscriptable<T>::value, void>::type> {
    std::size_t operator()(const T &p) const {
        using U = decltype(p[0]);
        return HashTable<Triplet<U>>()(Triplet<U>{p[0], p[1], p[2]});
    }
};

// Primary template for KeyEqual<T> using SFINAE
template <typename T, typename = void> struct KeyEqual {
    bool operator()(const T &t1, const T &t2) const { return t1 == t2; }
};

// Partial specialization for KeyEqual<Triplet<T>> when T is floating point type
// using SFINAE
template <typename T>
struct KeyEqual<Triplet<T>, typename std::enable_if<
                                std::is_floating_point<T>::value>::type> {
    bool operator()(const Triplet<T> &t1, const Triplet<T> &t2) const {
        return std::abs(std::get<0>(t1) - std::get<0>(t2)) < 1e-6 &&
               std::abs(std::get<1>(t1) - std::get<1>(t2)) < 1e-6 &&
               std::abs(std::get<2>(t1) - std::get<2>(t2)) < 1e-6;
    }
};

// Partial specialization for KeyEqual<Triplet<T>> when T is integer type using
// SFINAE
template <typename T>
struct KeyEqual<Triplet<T>,
                typename std::enable_if<std::is_integral<T>::value>::type> {
    bool operator()(const Triplet<T> &t1, const Triplet<T> &t2) const {
        return std::get<0>(t1) == std::get<0>(t2) &&
               std::get<1>(t1) == std::get<1>(t2) &&
               std::get<2>(t1) == std::get<2>(t2);
    }
};

// Partial specialization for KeyEqual<Point3D> using SFINAE
template <typename T> struct KeyEqual<T, typename std::enable_if<Subscriptable<T>::value>::type> {
    bool operator()(const T &p1, const T &p2) const {
        using U = decltype(p1[0]);
        return KeyEqual<Triplet<U>>()(
            Triplet<U>{p1[0], p1[1], p1[2]},
            Triplet<U>{p2[0], p2[1], p2[2]});
    }
};

template <typename Key, typename Value>
using UnorderedMap =
    std::unordered_map<Key, Value, HashTable<Key>, KeyEqual<Key>>;

template <typename Key>
using UnorderedSet = std::unordered_set<Key, HashTable<Key>, KeyEqual<Key>>;

} // namespace GeoRd::details


#endif // __UTILS_H__