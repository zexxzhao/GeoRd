#ifndef __POINT_H__
#define __POINT_H__
#include <array>
#include <cmath>
#include "MPI.h"


namespace GeoRd {

namespace details {
struct AddTag {};
struct SubTag {};
struct MulTag {};
struct DivTag {};

template <typename Tag, typename T, typename U, std::size_t N>
inline void recursive_operator(T a, const U b) {}

template <typename Tag, typename T, std::size_t N,
          std::enable_if_t<std::is_same_v<AddTag, Tag>, int> = 0>
inline void recursive_operator(T *a, const T *b) {
    if constexpr (N > 0) {
        a[0] += b[0];
        recursive_operator<Tag, T, N - 1>(a + 1, b + 1);
    }
}
template <typename Tag, typename T, std::size_t N,
          std::enable_if_t<std::is_same_v<SubTag, Tag>, int> = 0>
inline void recursive_operator(T *a, const T *b) {
    if constexpr (N > 0) {
        a[0] -= b[0];
        recursive_operator<Tag, T, N - 1>(a + 1, b + 1);
    }
}

template <typename Tag, typename T, std::size_t N,
          std::enable_if_t<std::is_same_v<MulTag, Tag>, int> = 0>
inline void recursive_operator(T *a, double alpha) {
    if constexpr (N > 0) {
        a[0] *= alpha;
        recursive_operator<Tag, T, N - 1>(a + 1, alpha);
    }
}
template <typename Tag, typename T, std::size_t N,
          std::enable_if_t<std::is_same_v<DivTag, Tag>, int> = 0>
inline void recursive_operator(T *a, double alpha) {
    recursive_operator<MulTag, T, N>(a + 1, 1.0 / alpha);
}
template <typename T, std::size_t N>
inline std::array<T, N> &operator+=(std::array<T, N> &a,
                                    const std::array<T, N> &b) {
    recursive_operator<AddTag, T, N>(a.data(), b.data());
    return a;
}
template <typename T, std::size_t N>
inline std::array<T, N> &operator-=(std::array<T, N> &a,
                                    const std::array<T, N> &b) {
    recursive_operator<SubTag, T, N>(a.data(), b.data());
    return a;
}

template <typename T, std::size_t N>
inline std::array<T, N> &operator*=(std::array<T, N> &a, double b) {
    recursive_operator<MulTag, T, N>(a.data(), b);
    return a;
}
template <typename T, std::size_t N>
inline std::array<T, N> &operator/=(std::array<T, N> &a, double b) {
    recursive_operator<DivTag, T, N>(a.data(), b);
    return a;
}
} // namespace details
template <int D = 3, std::enable_if_t<1 <= D and D <= 3, int> = 0> class Point {
public:
    using AtomizedType = double;

public:
    Point() : _x{} {}

    Point(const AtomizedType *x) { std::copy(x, x + D, _x.data()); }

    template <typename... Args, std::enable_if_t<sizeof...(Args) == D, int> = 0,
              std::enable_if_t<std::is_convertible_v<
                                   std::common_type_t<Args...>, AtomizedType>,
                               int> = 0>
    Point(Args... args) : _x{args...} {}

    const AtomizedType *coordinates() const { return _x.data(); }
    AtomizedType *coordinates() { return _x.data(); }

    AtomizedType x() const { return (*this)[0]; }
    AtomizedType y() const {
        if constexpr (D >= 2) {
            return (*this)[1];
        }
        else {
            error("A 1D Point does not have 2nd dimension.\n");
            return 0.0;
        }
    }
    AtomizedType z() const {
        if constexpr (D >= 3) {
            return (*this)[2];
        }
        else {
            error("A 1D or 2D Point does not have 3rd dimension.\n");
            return 0.0;
        }
    }

    AtomizedType operator[](size_t i) const { return this->_x[i]; }
    AtomizedType &operator[](size_t i) { return this->_x[i]; }

    Point operator+(const Point &other) const {
        auto p = *this;
        p += other;
        return p;
    }
    Point operator-(const Point &other) const {
        auto p = *this;
        p -= other;
        return p;
    }
    Point operator*(double alpha) const {
        auto p = *this;
        p *= alpha;
        return p;
    }
    Point operator/(double alpha) const {
        auto p = *this;
        p /= alpha;
        return p;
    }

    Point &operator+=(const Point &other) {
        using details::operator+=;
        _x += other._x;
        return *this;
    }
    Point &operator-=(const Point &other) {
        using details::operator-=;
        _x -= other._x;
        return *this;
    }
    Point &operator*=(double alpha) {
        using details::operator*=;
        _x *= alpha;
        return *this;
    }
    Point &operator/=(double alpha) {
        using details::operator/=;
        (*this) *= 1.0 / alpha;
        return *this;
    }

    AtomizedType dot(const Point &v) const {
        return std::inner_product(_x.begin(), _x.end(), v._x.begin(),
                                  static_cast<AtomizedType>(0.0));
    }
    Point cross(const Point &v) const {
        if constexpr (D == 3) {
            auto _y = v._x;
            double z[] = {0, 0, 0};
            z[0] = _x[1] * _y[2] - _x[2] * _y[1];
            z[1] = _x[2] * _y[0] - _x[0] * _y[2];
            z[2] = _x[0] * _y[1] - _x[1] * _y[0];
            return Point(z);
        }
        else {
            return {};
        }
    }
    AtomizedType norm() const { return std::sqrt(this->dot(*this)); }

    AtomizedType distance(const Point &v) const { return (*this - v).norm(); }

private:
    std::array<AtomizedType, D> _x;
};

using Point3D = Point<3>;

template <int D = 3> inline Point<D> operator*(double a, const Point<D> &p) {
    return p * a;
}

inline typename Point<3>::AtomizedType volume(const Point<3> &p0,
                                              const Point<3> &p1,
                                              const Point<3> &p2,
                                              const Point<3> &p3) {

    const auto ra = p1 - p0;
    const auto rb = p2 - p0;
    const auto rc = p3 - p0;
    double vol = 1.0 / 6.0 * ra.cross(rb).dot(rc);
    return std::abs(vol);
}

inline typename Point<3>::AtomizedType
area(const Point<3> &p0, const Point<3> &p1, const Point<3> &p2) {
    const auto ra = p1 - p0;
    const auto rb = p2 - p0;
    double area = 0.5 * ra.cross(rb).norm();
    return std::abs(area);
}

} // namespace GeoRd
#endif
