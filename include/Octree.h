#ifndef __OCTREE_H__
#define __OCTREE_H__
#include <algorithm>
#include <array>
#include <iostream>
#include <string>
#include <vector>

namespace GeoRd {

const int is_empty = static_cast<int>(-1);

template <int D> constexpr int n_child = 1 << D;

using Dim = std::integral_constant<int, 3>;

template <int D = Dim::value> struct Node {
    Node(int n = is_empty) { std::fill(vertex.begin(), vertex.end(), n); }

    void set_vertex(int i, int val) { this->vertex[i] = val; }

    int operator[](int i) const { return this->vertex[i]; }
    int &operator[](int i) { return this->vertex[i]; }

    [[deprecated]] void print_vertex() const {
        for (int i = 0; i < 8; i++) {
            std::cout << this->vertex[i] << std::endl;
        }
        std::cout << std::endl;
    }
    std::array<int, n_child<D>> vertex;
};

template<int D = Dim::value> struct TreeNode {
    std::vector<int> plist;
    std::vector<TreeNode> child;
    std::array<int, n_child<D>> child_exist;

    TreeNode(int n = is_empty) {
        std::fill(child_exist.begin(), child_exist.end(), n);
    }
};

#include "Point.h"

class Triangle3D {
public:
    std::array<Point3D, 3> vertex;
    Triangle3D(Point3D pa = {}, Point3D pb = {}, Point3D pc = {}) {
        this->vertex[0] = pa;
        this->vertex[1] = pb;
        this->vertex[2] = pc;
    }
    Triangle3D(double xa, double ya, double za, double xb, double yb, double zb,
               double xc, double yc, double zc)
        : Triangle3D(Point3D{xa, ya, za}, Point3D{xb, yb, zb},
                     Point3D{xc, yc, zc}) {}

    void print_tri() {
        std::cout << "Triangle3D Point 0: " << this->vertex[0].x() << " , "
                  << this->vertex[0].y() << " , " << this->vertex[0].z()
                  << std::endl;
        std::cout << "Triangle3D Point 1: " << this->vertex[1].x() << " , "
                  << this->vertex[1].y() << " , " << this->vertex[1].z()
                  << std::endl;
        std::cout << "Triangle3D Point 2: " << this->vertex[2].x() << " , "
                  << this->vertex[2].y() << " , " << this->vertex[2].z()
                  << std::endl;
    }
};

struct Box3D {
    std::array<Point3D, 2> box_pos;

    Box3D(Point3D pa = {}, Point3D pb = {}) : box_pos{pa, pb} {}
    Box3D(double xa, double ya, double za, double xb, double yb, double zb)
        : Box3D(Point3D{xa, ya, za}, Point3D{xb, yb, zb}) {}

    Box3D &operator=(Box3D box) {
        std::swap(this->box_pos, box.box_pos);
        return *this;
    }

    [[deprecated]] void set_point(Point3D pa, Point3D pb) {
        this->box_pos[0] = pa;
        this->box_pos[1] = pb;
    }

    Point3D operator[](int i) const { return this->box_pos[i]; }
    Point3D &operator[](int i) { return this->box_pos[i]; }

    [[deprecated]] void print_box() const {
        std::cout << "Box Point3D 0: " << this->box_pos[0].x() << " , "
                  << this->box_pos[0].y() << " , " << this->box_pos[0].z()
                  << std::endl;
        std::cout << "Box Point3D 1: " << this->box_pos[1].x() << " , "
                  << this->box_pos[1].y() << " , " << this->box_pos[1].z()
                  << std::endl;
        std::cout << std::endl;
    }
};

namespace details {

// build_bounding_box: build the bounding box of the point cloud
// begin: begin iterator of the point cloud
// end: end iterator of the point cloud
// box_tol: tolerance of the bounding box
template <typename PointIT>
Box3D build_bounding_box(PointIT begin, PointIT end, double box_tol = 1e-6) {

    // get the maximum x coordinates of the point cloud
    auto xmax =
        std::max_element(begin, end, [](const Point3D &a, const Point3D &b) {
            return a[0] < b[0];
        });
    // get the maximum y coordinates of the point cloud
    auto ymax =
        std::max_element(begin, end, [](const Point3D &a, const Point3D &b) {
            return a[1] < b[1];
        });
    // get the maximum z coordinates of the point cloud
    auto zmax =
        std::max_element(begin, end, [](const Point3D &a, const Point3D &b) {
            return a[2] < b[2];
        });
    // get the minimum x coordinates of the point cloud
    auto xmin =
        std::min_element(begin, end, [](const Point3D &a, const Point3D &b) {
            return a[0] < b[0];
        });
    // get the minimum y coordinates of the point cloud
    auto ymin =
        std::min_element(begin, end, [](const Point3D &a, const Point3D &b) {
            return a[1] < b[1];
        });
    // get the minimum z coordinates of the point cloud
    auto zmin =
        std::min_element(begin, end, [](const Point3D &a, const Point3D &b) {
            return a[2] < b[2];
        });

    // build the bounding box
    Point3D p0{(*xmin)[0] - box_tol, (*ymin)[1] - box_tol, (*zmin)[2] - box_tol};
    Point3D p1{(*xmax)[0] + box_tol, (*ymax)[1] + box_tol, (*zmax)[2] + box_tol};
    return Box3D{p0, p1};
}

inline Box3D get_tri_bounding_box(Triangle3D tri) {
    return build_bounding_box(tri.vertex.begin(), tri.vertex.end(), 1e-5);
}

inline bool box_box_intersect(const Box3D &box1, const Box3D &box2) {
    Point3D box1_center;
    Point3D box2_center;
    Point3D wid_avg;
    Point3D dist_abs;

    box1_center = (box1.box_pos[1] + box1.box_pos[0]) / 2.0;
    box2_center = (box2.box_pos[1] + box2.box_pos[0]) / 2.0;
    wid_avg = (box1.box_pos[1] - box1.box_pos[0]) / 2.0 +
              (box2.box_pos[1] - box2.box_pos[0]) / 2.0;
    dist_abs = box2_center - box1_center;

    for (int i = 0; i < 3; i++) {
        if (std::abs(dist_abs[i]) > wid_avg[i]) {
            return false;
        }
    }
    return true;
}

inline Box3D cal_sub_box(const Box3D &box, int nk) {
    int b[3];
    b[2] = (nk) / 4;
    b[1] = (nk - b[2] * 4) / 2;
    b[0] = (nk - b[2] * 4 - b[1] * 2) / 1;

    Point3D pa = box.box_pos[0];
    Point3D pb = (box.box_pos[1] + box.box_pos[0]) / 2.0;
    Point3D diag_vec = (box.box_pos[1] - box.box_pos[0]) / 2.0;

    Point3D ra;
    Point3D rb;
    for (int i = 0; i < 3; i++) {
        ra[i] = pa[i] * b[i] + pb[i] * (1.0 - b[i]);
    }
    rb = ra + diag_vec;
    Box3D box_res(ra, rb);
    return box_res;
}

inline bool point_in_box(const Point3D &pcur, const Box3D &box) {
    for (int i = 0; i < 3; i++) {
        if (pcur[i] < box.box_pos[0][i] or pcur[i] > box.box_pos[1][i]) {
            return false;
        }
    }
    return true;
}

inline double compute_min_dist(Box3D box, Point3D x) {
    double r2 = 0.0;
    double b[6];

    b[0] = box.box_pos[0][0];
    b[1] = box.box_pos[0][1];
    b[2] = box.box_pos[0][2];

    b[3] = box.box_pos[1][0];
    b[4] = box.box_pos[1][1];
    b[5] = box.box_pos[1][2];

    if (x[0] < b[0])
        r2 += (x[0] - b[0]) * (x[0] - b[0]);
    if (x[0] > b[3])
        r2 += (x[0] - b[3]) * (x[0] - b[3]);
    if (x[1] < b[1])
        r2 += (x[1] - b[1]) * (x[1] - b[1]);
    if (x[1] > b[4])
        r2 += (x[1] - b[4]) * (x[1] - b[4]);
    if (x[2] < b[2])
        r2 += (x[2] - b[2]) * (x[2] - b[2]);
    if (x[2] > b[5])
        r2 += (x[2] - b[5]) * (x[2] - b[5]);

    r2 = std::sqrt(r2);

    return r2;
}

} // namespace details


template <typename = void, int = 0> struct Octree {};

// for point
template <int _Tag> struct Octree<Point3D, _Tag> {
    TreeNode<3> root{};
    std::vector<Point3D> pdat;
    Box3D box_coord;
    int recur_lim = 6;

    Octree() = default;

    void set_box(std::vector<Point3D> pvec) {
        details::build_bounding_box(pvec.begin(), pvec.end(), 1e-5);
    }

    void insert_point(const std::vector<Point3D> &plist) {
        Box3D box_cur = this->box_coord;
        for (int i = 0; i < plist.size(); i++) {
            insert_point_help(plist[i], i, 0, box_cur, this->root);
        }
    }

    void insert_point_help(Point3D pcur, int point_id, int level, Box3D box_cur,
                           TreeNode<3> &cur) {
        int is_in_box;
        Box3D box_sub;

        if (level == this->recur_lim) {
            cur.plist.push_back(point_id);
            return;
        }

        is_in_box = details::point_in_box(pcur, box_cur);

        if (is_in_box == 1) {

            if (cur.child.size() == 0) {
                for (int i = 0; i < 8; i++) {
                    cur.child.push_back(TreeNode());
                }
            }

            if (cur.child.size() != 8) {
                std::cout << "Child size wrong" << std::endl;
            }

            for (int i = 0; i < 8; i++) {
                box_sub = details::cal_sub_box(box_cur, i);
                is_in_box = details::point_in_box(pcur, box_sub);

                if (is_in_box == 1) {
                    cur.child_exist[i] = 1;
                    insert_point_help(pcur, point_id, level + 1, box_sub,
                                      cur.child[i]);
                    break;
                }
            }
        }
    }
    void search_point(Point3D pcur, int &index, double &rval) {
        index = 0;
        rval = (pcur - this->pdat[0]).norm();
        search_point_help(pcur, 0, this->root, this->box_coord, index, rval);
    }

    void search_point_help(Point3D pcur, int level, TreeNode<3> &cur,
                           Box3D box_cur, int &index, double &rval) {
        Box3D box_sub;

        if (level == this->recur_lim) {
            for (int i = 0; i < cur.plist.size(); i++) {
                double dist_val = (this->pdat[cur.plist[i]] - pcur).norm();
                if (dist_val < rval) {
                    rval = dist_val;
                    index = cur.plist[i];
                }
            }
        }
        else {
            for (int i = 0; i < 8; i++) {
                if (cur.child_exist[i] == -1)
                    continue;

                box_sub = details::cal_sub_box(box_cur, i);
                double dist_tmp = details::compute_min_dist(box_sub, pcur);

                if (dist_tmp > rval)
                    continue;
                search_point_help(pcur, level + 1, cur.child[i], box_sub, index,
                                  rval);
            }
        }
    }

    [[deprecated]] void print_octree_box() const {
        std::cout << "Octree box Point 0: " << this->box_cor.box_pos[0].x()
                  << " , " << this->box_cor.box_pos[0].y() << " , "
                  << this->box_cor.box_pos[0].z() << std::endl;
        std::cout << "Octree box Point 1: " << this->box_cor.box_pos[1].x()
                  << " , " << this->box_cor.box_pos[1].y() << " , "
                  << this->box_cor.box_pos[1].z() << std::endl;
    }
};

#include <unordered_map>

// for triangles
template <int _Tag> class Octree<Triangle3D, _Tag> {
    Box3D box_cor;
    const int recur_lim = 6;
    std::vector<Box3D> node_box;
    std::vector<Node<3>> node_dat;
    std::vector<Box3D> entity_box;
    std::vector<std::vector<int>> entity_bucket;
    std::unordered_map<std::string, int> path_hash;

    Octree() : node_dat(1) {}

    void set_box(Point3D pa, Point3D pb) {
        this->box_cor.box_pos[0] = pa;
        this->box_cor.box_pos[1] = pb;
    }

    void insert_triangle(Triangle3D tri, int tri_id) {
        int level = 0;
        std::string path = "";
        auto tri_box = details::get_tri_bounding_box(tri);
        recur_insert(tri_id, tri_box, 0, this->box_cor, level, path);
    }
    void recur_insert(int tri_id, Box3D tri_box, int node_num, Box3D oct_box,
                      int level, std::string path) {

        Box3D box_sub[8];
        bool travel[8];

        // oct_box.print_box();

        if (level >= recur_lim) {
            auto hash_sear = path_hash.find(path);

            if (hash_sear == (path_hash.end())) {
                path_hash[path] = entity_bucket.size();

                std::vector<int> tmp;
                tmp.push_back(tri_id);
                entity_bucket.push_back(tmp);

                entity_box.push_back(oct_box);
            }
            else {
                entity_bucket[hash_sear->second].push_back(tri_id);
            }

            return;
        }

        for (int isub = 0; isub < 8; isub++) {

            box_sub[isub] = details::cal_sub_box(oct_box, isub);
            travel[isub] = details::box_box_intersect(tri_box, box_sub[isub]);

            if (node_dat[node_num].vertex[isub] == -1 and
                travel[isub] == true) {
                Node node_insert;
                node_dat.push_back(node_insert);
                node_box.push_back(box_sub[isub]);

                int node_insert_id = (node_dat.size()) - 1;
                node_dat[node_num].vertex[isub] = node_insert_id;
            }
        }

        for (int isub = 0; isub < 8; isub++) {
            if (travel[isub] == true) {
                std::string path_next = path + std::to_string(isub);
                recur_insert(tri_id, tri_box, node_dat[node_num].vertex[isub],
                             box_sub[isub], level + 1, path_next);
            }
        }
    }

    void print_octree_box() {
        std::cout << "Octree box Point 0: " << this->box_cor.box_pos[0].x()
                  << " , " << this->box_cor.box_pos[0].y() << " , "
                  << this->box_cor.box_pos[0].z() << std::endl;
        std::cout << "Octree box Point 1: " << this->box_cor.box_pos[1].x()
                  << " , " << this->box_cor.box_pos[1].y() << " , "
                  << this->box_cor.box_pos[1].z() << std::endl;
    }
};


} // namespace GeoRd
#endif // __OCTREE_H__