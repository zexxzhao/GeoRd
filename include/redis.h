#ifndef __REDIS_H__
#define __REDIS_H__
#include "Mesh.h"
#include "Octree.h"
#include <set>
#include <unorder_map>
namespace GeoRd {

using TetrahedronMesh = Mesh<Element::Tetrahedron, 3, Layout::Distributed>;

using TriangleMesh = Mesh<Element::Triangle, 3, Layout::Exclusive>;

namespace details {
template <typename T, typename U>
void write_interface_txt(const std::vector<T> &vx, const std::vector<T> &vy,
                         const std::vector<T> &vz,
                         const std::vector<U> &color) {
    std::string interface_name = "phi0/tri.dat";
    std::ofstream stFile(interface_name);

    for (int i = 0; i < vx.size(); i++) {
        stFile << std::fixed << std::setprecision(10) << vx[i] << " " << vy[i]
               << " " << vz[i] << " " << color[i] << "\n";
    }
    stFile.close();
}

// Write out phi=0 surface triangulation
template <typename T, typename U>
void write_triangulation(
    const std::vector<T> &vx_res, const std::vector<T> &vy_res,
    const std::vector<T> &vz_res, const std::vector<T> &nx_res,
    const std::vector<T> &ny_res, const std::vector<T> &nz_res,
    const std::vector<U> &c1_res, const std::vector<U> &c2_res,
    const std::vector<U> &c3_res) {
    int rankvalues = details::MPI_rank(MPI_COMM_WORLD);

    if (rankvalues == 0) {
        std::string name = "sur/vertex.dat";
        std::ofstream pFile(name);
        for (int itmp = 0; itmp < vx_res.size(); itmp++) {
            pFile << std::fixed << std::setprecision(10) << vx_res[itmp] << " "
                  << vy_res[itmp] << " " << vz_res[itmp] << "\n";
        }
        pFile.close();
    }

    if (rankvalues == 0) {
        std::string name = "sur/normal.dat";
        std::ofstream pFile(name);
        for (int itmp = 0; itmp < nx_res.size(); itmp++) {
            pFile << std::fixed << std::setprecision(10) << nx_res[itmp] << " "
                  << ny_res[itmp] << " " << nz_res[itmp] << "\n";
        }
        pFile.close();
    }

    if (rankvalues == 0) {
        std::string name = "sur/connect.dat";
        std::ofstream pFile(name);
        for (int itmp = 0; itmp < c1_res.size(); itmp++) {
            pFile << c1_res[itmp] << " " << c2_res[itmp] << " " << c3_res[itmp]
                  << "\n";
        }
        pFile.close();
    }
}

/**
 * @brief Make sure the normal vector of the triangle is pointing where phi
 * is positive
 *
 * @param[in] phi_vertex vertex coordinate of the triangulation mesh
 * @param[in] phi_tet_vertex vertex coordinate of the tetrahedral
 * @param[out] tri_con_tmp connectivity list of the triangle
 * @param[in] phival scalar function value of the tetrahedral mesh
 */
void compute_triangle_winding_in_tetrahedron(
    const std::vector<Point3D> &phi_vertex,
    const std::vector<Point3D> &phi_tet_vertex, std::vector<int> &tri_con_tmp,
    const std::array<double, typename VolumeMesh::Element::n_vertices>
        &phival) {

    // Get triangle vertex coordinate
    auto v1_tmp = phi_vertex[tri_con_tmp[0]];
    v2_tmp = phi_vertex[tri_con_tmp[1]];
    v3_tmp = phi_vertex[tri_con_tmp[2]];

    // Calculate the normal vector for triangle
    const double unit_scale = 1e6;
    auto e12 = (v2_tmp - v1_tmp) * unit_scale;
    auto e13 = (v3_tmp - v1_tmp) * unit_scale;
    auto nor = e12.cross(e13);
    nor /= nor.norm() + 1e-30;

    // Check the direction of normal
    // get the index of the vertex with largest absolute phi value
    auto it =
        std::max_element(phival.begin(), phival.end(), [](double a, double b) {
            return std::abs(a) < std::abs(b);
        });
    auto sign = (*it > 0 ? 1 : -1);
    auto index = std::distance(phival.begin(), it);

    auto dot_val = (phi_tet_vertex[index] - v1_tmp).dot(nor) * sign;

    if (dot_val < 0.0) {
        std::swap(tri_con_tmp[1], tri_con_tmp[2]);
    }
}

// For 4 Node case
// ----
// |\/|
// |/\|
// ----
// Need to choose \ or / direction
int compute_diagonal_direction(std::vector<Point> &phi_tmp_vertex) {
    int diag_dir = 1;

    Point3D v1 = phi_tmp_vertex[1] - phi_tmp_vertex[0];
    Point3D v2 = phi_tmp_vertex[2] - phi_tmp_vertex[0];
    Point3D v3 = phi_tmp_vertex[3] - phi_tmp_vertex[0];

    double cos12 = details::cos(v1, v2);
    double cos13 = details::cos(v1, v3);
    double cos23 = details::cos(v2, v3);

    if (cos12 <= cos13 and cos12 <= cos23) {
        diag_dir = 3;
    }
    if (cos13 <= cos12 and cos13 <= cos23) {
        diag_dir = 2;
    }
    if (cos23 <= cos12 and cos23 <= cos13) {
        diag_dir = 1;
    }

    return diag_dir;
}

// Adding triangle
// 1 triangle or 2 triangles
template <typename VolumeMesh>
void add_triangle(const VolumeMesh &mesh, const std::vector<double> &phid0,
                  std::vector<Point3D> &phi_vertex,
                  std::vector<std::vector<int>> &phi_connect) {
    // std::vector<Point>             phi_tmp_vertex;
    // std::vector<Point>             phi_tet_vertex;

    // std::vector<double> coordinate_dofs;
    // std::vector<int> phi_pos;
    // std::vector<int> phi_neg;

    // Array<double> vcor(3);
    // Array<double> phival(4);
    // Array<double> phival_vert(1);

    // double pos_cor[3];
    // double neg_cor[3];
    // double phi_cor[3];

    // std::string hash_tmp;

    int ver_loc_index_cur = 0;
    // std::unordered_map<std::string, int> vertex_to_local_index;
    details::Point3DMap<std::size_t> vertex_to_local_index;

    // for (CellIterator cell(*mesh); !cell.end(); ++cell)
    for (int icell = 0; icell < mesh.elements.size() / 4; ++icell) {
        // clear vector
        // std::vector<Point>().swap(phi_tmp_vertex);
        // std::vector<Point>().swap(phi_tet_vertex);
        // std::vector<int>().swap(phi_pos);
        // std::vector<int>().swap(phi_neg);
        std::vector<Point3D> phi_tmp_vertex;
        std::array<Point3D, typename VolumeMesh::Element::n_vertices>
            phi_tet_vertex;
        std::vector<int> phi_pos, phi_neg;

        // Get the coordinate of four vertex
        // mesh.get_cell_coordinates(icell, coordinate_dofs);
        mesh.get_cell_vertices(icell, phi_tet_vertex);
        // Get the phi value of four vertex
        std::array<double, typename VolumeMesh::Element::n_vertices> phival;
        for (int ivtx = 0; ivtx < typename VolumeMesh::Element::n_vertices;
             ivtx++) {
            phival[ivtx] = phid0[mesh.cells[icell][ivtx]];
            if (phival[ivtx] > 0.0) {
                phi_pos.push_back(ivtx);
            }
            else {
                phi_neg.push_back(ivtx);
            }
        }

        // save phi = 0 vertex (3 or 4 points)
        for (int ipos = 0; ipos < phi_pos.size(); ipos++) {
            for (int ineg = 0; ineg < phi_neg.size(); ineg++) {
                // Use linear interpolation to get the phi=0 coordinate
                auto ratio =
                    (0.0 - phival[phi_neg[ineg]]) /
                    (phival[phi_pos[ipos]] - phival[phi_neg[ineg]] + 1e-30);
                auto phi_coord = (1.0 - ratio) * phi_tet_vertex[phi_neg[ineg]] +
                                 ratio * phi_tet_vertex[phi_pos[ipos]];

                // Confusion: Why need to clip the data?
                // Number near zero will cause problem in hash function
                // Update: New version of hash function does not need to
                // clip the data
                // clip the data
                double clip_tol = 1e-10;
                for (int ic = 0; ic < 3; ic++) {
                    if (std::abs(phi_cor[ic]) < clip_tol) {
                        phi_cor[ic] = 0.0;
                    }
                }

                // save phi=0 coordinate
                phi_tmp_vertex.push_back(phi_coord);
            }
        }

        // 3 points case (Need to add one triangle)
        if (phi_tmp_vertex.size() == 3) {
            assert(phi_pos.size() == 1 or phi_neg.size() == 1);
            // Make sure it is empty
            std::vector<int> tri_con_tmp;

            for (int itmp = 0; itmp < phi_tmp_vertex.size(); itmp++) {

                // if the vertex is already in the vector, just push back
                // the index else, push back the vertex and the index
                if (vertex_to_local_index.find(phi_tmp_vertex[itmp]) !=
                    vertex_to_local_index.end()) {
                    tri_con_tmp.push_back(
                        vertex_to_local_index[phi_tmp_vertex[itmp]]);
                }
                else {
                    phi_vertex.push_back(phi_tmp_vertex[itmp]);
                    tri_con_tmp.push_back(phi_vertex.size() - 1);
                    vertex_to_local_index[phi_tmp_vertex[itmp]] =
                        ver_loc_index_cur;
                    ver_loc_index_cur++;
                }
            }

            // if the triangle is degenerated, skip it
            if (tri_con_tmp[0] == tri_con_tmp[1] or
                tri_con_tmp[0] == tri_con_tmp[2] or
                tri_con_tmp[1] == tri_con_tmp[2]) {
                continue;
            }

            // Fix winding problem
            compute_triangle_winding_in_tetrahedron(phi_vertex, phi_tet_vertex,
                                                    tri_con_tmp, phival);

            // add triangle
            phi_connect.push_back(tri_con_tmp);

            // debug
            /*
            if(rankvalues==0){
                std::cout << "v1-x:"  << e12.x() << " y:" << e12.y() << "
            z:" << e12.z() << std::endl; std::cout << "v2-x:"  << e13.x() <<
            " y:" << e13.y() << " z:" << e13.z() << std::endl; std::cout <<
            "nor-x:" << nor.x() << " y:" << nor.y() << " z:" << nor.z() <<
            std::endl; std::cout << "vcheck-x:" << vcheck_tmp.x() << " y:"
            << vcheck_tmp.y() << " z:" << vcheck_tmp.z() << std::endl;
                std::cout << "dot_val:"  << dot_val << std::endl;
                std::cout << std::endl;
            }
            */
        }

        // 4 points case (Need to add two triangle)
        else if (phi_tmp_vertex.size() == 4) {
            assert(phi_pos.size() == 2 and phi_neg.size() == 2);
            std::array<int, 4> vertex_4_index{};

            for (int itmp = 0; itmp < phi_tmp_vertex.size(); itmp++) {
                if (vertex_to_local_index.find(phi_tmp_vertex[itmp]) !=
                    vertex_to_local_index.end()) {
                    vertex_4_index[itmp] =
                        vertex_to_local_index[phi_tmp_vertex[itmp]];
                }
                else {
                    phi_vertex.push_back(phi_tmp_vertex[itmp]);
                    vertex_4_index[itmp] = phi_vertex.size() - 1;
                    vertex_to_local_index[phi_tmp_vertex[itmp]] =
                        ver_loc_index_cur;
                    ver_loc_index_cur++;
                }
            }

            int tmp_mark = compute_diagonal_direction(phi_tmp_vertex);

            // Add two triangle here
            for (int itmp = 1; itmp < 4; itmp++) {
                if (itmp == tmp_mark) {
                    continue;
                }
                // Make sure they are empty
                std::vector<int> tri_con_tmp;

                // vertex 0
                tri_con_tmp.push_back(vertex_4_index[0]);
                // vertex tmp_mark
                tri_con_tmp.push_back(vertex_4_index[tmp_mark]);
                // vertex itmp
                tri_con_tmp.push_back(vertex_4_index[itmp]);

                // if the triangle is degenerated, skip it
                if (tri_con_tmp[0] == tri_con_tmp[1] or
                    tri_con_tmp[0] == tri_con_tmp[2] or
                    tri_con_tmp[1] == tri_con_tmp[2]) {
                    continue;
                }

                // Fix winding problem
                compute_triangle_winding_in_tetrahedron(
                    phi_vertex, phi_tet_vertex, tri_con_tmp, phival);

                // add triangle
                phi_connect.push_back(tri_con_tmp);
            }
        }
    }
}

// clean triangle
void clean_triangle(const std::vector<int> &c1_tmp,
                    const std::vector<int> &c2_tmp,
                    const std::vector<int> &c3_tmp, std::vector<int> &c1_res,
                    std::vector<int> &c2_res, std::vector<int> &c3_res,
                    const std::vector<double> &vx_un_res,
                    const std::vector<double> &vy_un_res,
                    const std::vector<double> &vz_un_res,
                    std::vector<double> &vx_res, std::vector<double> &vy_res,
                    std::vector<double> &vz_res) {
    std::vector<int> c1_un_res;
    std::vector<int> c2_un_res;
    std::vector<int> c3_un_res;

    std::vector<int> color;
    std::vector<int> color_num;
    std::vector<int> ver_map;
    ver_map.resize(vx_un_res.size());

    int con_glo_index_cur = 0;
    std::string hash_tmp;
    // std::unordered_map<std::string, int> con_glo_index_map;
    details::UnorderedMap<details::Triplet<int>, std::size_t> con_glo_index_map;

    int rankvalues = details::MPI_rank(MPI_COMM_WORLD);

    // Triangle cleaning
    if (rankvalues == 0) {
        // Map triangle to global index
        for (int itmp = 0; itmp < c1_tmp.size(); itmp++) {

            if (c1_tmp[itmp] == c2_tmp[itmp] or c1_tmp[itmp] == c3_tmp[itmp] or
                c2_tmp[itmp] == c3_tmp[itmp]) {
                continue;
            }
            details::Triplet<int> tri_tmp{c1_tmp[itmp], c2_tmp[itmp],
                                          c3_tmp[itmp]};
            auto it = con_glo_index_map.find(tri_tmp);
            if (it == con_glo_index_map.end()) {
                c1_un_res.push_back(c1_tmp[itmp]);
                c2_un_res.push_back(c2_tmp[itmp]);
                c3_un_res.push_back(c3_tmp[itmp]);
                con_glo_index_map.insert({tri_tmp, con_glo_index_cur});
                con_glo_index_cur++;
            }
        }

        // cal_color(node_connect, color, color_num);
        for (int i = 0; i < color_num.size(); i++) {
            std::cout << "    Surface color " << i
                      << " number: " << color_num[i] << std::endl;
        }
        // write_interface_txt(vx_un_res, vy_un_res, vz_un_res, color);

        int num_tol = 50;
        int clean_index = 0;
        for (int i = 0; i < node_connect.size(); i++) {
            int color_nk = color_num[color[i] - 1];
            // std::cout<<"Index "<<i<<" number: "<<color_nk<<std::endl;
            if (color_nk >= num_tol) {
                vx_res.push_back(vx_un_res[i]);
                vy_res.push_back(vy_un_res[i]);
                vz_res.push_back(vz_un_res[i]);
                ver_map[i] = clean_index;
                clean_index++;
            }
            else {
                ver_map[i] = -1;
            }
        }

        for (int iele = 0; iele < c1_un_res.size(); iele++) {
            int c1_index = c1_un_res[iele];
            int c2_index = c2_un_res[iele];
            int c3_index = c3_un_res[iele];

            int v1_vermap_val = ver_map[c1_index];
            int v2_vermap_val = ver_map[c2_index];
            int v3_vermap_val = ver_map[c3_index];

            if (v1_vermap_val == -1 or v2_vermap_val == -1 or
                v3_vermap_val == -1) {
                continue;
            }
            else {
                c1_res.push_back(v1_vermap_val);
                c2_res.push_back(v2_vermap_val);
                c3_res.push_back(v3_vermap_val);
            }
        }
    }

    details::MPI_broadcast(MPI_COMM_WORLD, vx_res);
    details::MPI_broadcast(MPI_COMM_WORLD, vy_res);
    details::MPI_broadcast(MPI_COMM_WORLD, vz_res);

    details::MPI_broadcast(MPI_COMM_WORLD, c1_res);
    details::MPI_broadcast(MPI_COMM_WORLD, c2_res);
    details::MPI_broadcast(MPI_COMM_WORLD, c3_res);
}

template <typename T>
void get_duplicated_triangles(const TriangleMesh &mesh,
                              std::vector<T> &duplicated_triangle_index) {
    using Triangle = details::Triplet<std::size_t>;
    std::unordered_set<Triangle, details::HashTable<Triangle>,
                       details::KeyEqual<Triangle>>
        triangles;

    for (std::size_t i = 0; i < mesh.elements.size();
         i += typename TriangleMesh::Element::n_vertices) {
        Triangle tri_tmp{mesh.elements[i], mesh.elements[i + 1],
                         mesh.elements[i + 2]};
        if (triangles.find(tri_tmp) == triangles.end()) {
            triangles.insert(tri_tmp);
        }
        else {
            duplicated_triangle_index.push_back(i);
        }
    }
}

template <typename T>
void get_small_droplets(const TriangleMesh &mesh,
                        std::vector<T> &small_droplets_index, int criteria) {
    // generate vertex connectivity
    Graph graph;
    get_vertex_connectivity(mesh, graph);

    // traverse the graph and color the vertices
    std::vector<std::vector<size_t>> color;
    connected_components(graph, color);

    // remove small droplets
    std::for_each(
        color.begin(), color.end(), [&](std::vector<size_t> &color_i) {
            if (color_i.size() < criteria) {
                small_droplets_index.insert(small_droplets_index.end(),
                                            color_i.begin(), color_i.end());
            }
        });
}

// remove duplicated triangles and small droplets
inline void clean_triangle(TriangleMesh &mesh, int criteria) {
    if (details::MPI_rank()) {
        return;
    }
    std::vector<std::size_t> removable_triangle_index;
    get_duplicated_triangles(mesh, removable_triangle_index);
    get_small_droplets(mesh, removable_triangle_index, criteria);
    std::set<std::size_t> removable_triangle_index_set(
        removable_triangle_index.begin(), removable_triangle_index.end());
    // copy the remaining triangles to a new mesh
    TriangleMesh mesh_clean;
    const auto nvtx_triangle = typename TriangleMesh::Element::n_vertices;
    mesh_clean.elements.reserve(
        mesh.elements.size() - nvtx_triangle * removable_triangle_index.size());
    for (std::size_t i = 0; i < mesh.elements.size(); i += nvtx_triangle) {
        if (removable_triangle_index_set.find(i) ==
            removable_triangle_index_set.end()) {
            mesh_clean.elements.push_back(mesh.elements[i]);
            mesh_clean.elements.push_back(mesh.elements[i + 1]);
            mesh_clean.elements.push_back(mesh.elements[i + 2]);
        }
    }
    // collect the remaining vertices
    std::set<std::size_t> remaining_vertices;
    for (std::size_t i = 0; i < mesh_clean.elements.size(); i++) {
        remaining_vertices.insert(mesh_clean.elements[i]);
    }
    // copy the remaining vertices to a new mesh
    mesh_clean.vertices.reserve(remaining_vertices.size());
    for (std::size_t i = 0; i < mesh.vertices.size(); i++) {
        if (remaining_vertices.find(i) != remaining_vertices.end()) {
            mesh_clean.vertices.push_back(mesh.vertices[i]);
        }
    }
    // update the vertex indices in the elements
    std::unordered_map<std::size_t, std::size_t> vertex_index_map;
    std::size_t new_index = 0;
    for (std::size_t i = 0; i < mesh_clean.elements.size(); i++) {
        auto it = vertex_index_map.find(mesh_clean.elements[i]);
        if (it == vertex_index_map.end()) {
            vertex_index_map[mesh_clean.elements[i]] = new_index;
            mesh_clean.elements[i] = new_index;
            new_index++;
        }
        else {
            mesh_clean.elements[i] = it->second;
        }
    }
    // update the mesh
    mesh = std::move(mesh_clean);
}

// combine triangles
void combine_triangle(const std::vector<Point3D> &phi_vertex,
                      const std::vector<std::vector<int>> &phi_connect,
                      std::vector<double> &vx_res, std::vector<double> &vy_res,
                      std::vector<double> &vz_res, std::vector<int> &c1_res,
                      std::vector<int> &c2_res, std::vector<int> &c3_res) {
    std::string hash_tmp;
    int ver_glo_index_cur = 0;

    // std::unordered_map<std::string, int> ver_glo_index_map;
    details::Point3DMap<size_t> ver_glo_index_map;

    std::vector<double> vx_un_res;
    std::vector<double> vy_un_res;
    std::vector<double> vz_un_res;

    std::vector<int> c1_tmp;
    std::vector<int> c2_tmp;
    std::vector<int> c3_tmp;

    // Following code remove the same vertex over different processor
    // Make sure they are unique
    // MPI communication:
    std::vector<double> phi_cx_loc;
    std::vector<double> phi_cy_loc;
    std::vector<double> phi_cz_loc;
    std::vector<double> phi_cx_glo;
    std::vector<double> phi_cy_glo;
    std::vector<double> phi_cz_glo;

    // copy to local vector
    for (int iphi = 0; iphi < phi_vertex.size(); iphi++) {
        phi_cx_loc.push_back(phi_vertex[iphi].x());
        phi_cy_loc.push_back(phi_vertex[iphi].y());
        phi_cz_loc.push_back(phi_vertex[iphi].z());
    }

    // MPI communication
    // MPI::gather(MPI_COMM_WORLD,phi_cx_loc,phi_cx_glo);
    // MPI::gather(MPI_COMM_WORLD,phi_cy_loc,phi_cy_glo);
    // MPI::gather(MPI_COMM_WORLD,phi_cz_loc,phi_cz_glo);

    // MPI::broadcast(MPI_COMM_WORLD,phi_cx_glo);
    // MPI::broadcast(MPI_COMM_WORLD,phi_cy_glo);
    // MPI::broadcast(MPI_COMM_WORLD,phi_cz_glo);
    details::MPI_allgatherv(MPI_COMM_WORLD, phi_cx_loc, phi_cx_glo);
    details::MPI_allgatherv(MPI_COMM_WORLD, phi_cy_loc, phi_cy_glo);
    details::MPI_allgatherv(MPI_COMM_WORLD, phi_cz_loc, phi_cz_glo);

    for (int itmp = 0; itmp < phi_cx_glo.size(); itmp++) {
        Point3D point{phi_cx_glo[itmp], phi_cy_glo[itmp], phi_cz_glo[itmp]};
        auto it = ver_glo_index_map.find(point);
        if (it == ver_glo_index_map.end()) {
            ver_glo_index_map[point] = ver_glo_index_cur;
            ver_glo_index_cur++;

            // Add new vertex to global vector
            vx_un_res.push_back(phi_cx_glo[itmp]);
            vy_un_res.push_back(phi_cy_glo[itmp]);
            vz_un_res.push_back(phi_cz_glo[itmp]);
        }
        // hash_cor(phi_cx_glo[itmp],phi_cy_glo[itmp],phi_cz_glo[itmp],hash_tmp);
        // auto hash_sear = ver_glo_index_map.find(hash_tmp);
        // //std::cout << hash_tmp << std::endl;
        // if( hash_sear == ver_glo_index_map.end() ){
        //     ver_glo_index_map[hash_tmp] = ver_glo_index_cur;
        //     ver_glo_index_cur++;

        //     // Add new vertex to global vector
        //     vx_un_res.push_back( phi_cx_glo[itmp] );
        //     vy_un_res.push_back( phi_cy_glo[itmp] );
        //     vz_un_res.push_back( phi_cz_glo[itmp] );
        // }
    }

    // Following code do MPI communication and change the old index in
    // connect to new index
    std::vector<int> tri_a_index_loc;
    std::vector<int> tri_b_index_loc;
    std::vector<int> tri_c_index_loc;
    std::vector<int> tri_a_index_tmp;
    std::vector<int> tri_b_index_tmp;
    std::vector<int> tri_c_index_tmp;

    // copy to local vector
    for (int iphi = 0; iphi < phi_connect.size(); iphi++) {
        tri_a_index_loc.push_back(phi_connect[iphi][0]);
        tri_b_index_loc.push_back(phi_connect[iphi][1]);
        tri_c_index_loc.push_back(phi_connect[iphi][2]);
    }

    // transform old index to new index on each local processor first
    for (int itmp = 0; itmp < tri_a_index_loc.size(); itmp++) {
        auto it = ver_glo_index_map.find(phi_vertex[tri_a_index_loc[itmp]]);
        if (it != ver_glo_index_map.end()) {
            tri_a_index_tmp.push_back(it->second);
        }
        else {
            printf("Error: can not find the vertex in local processor");
        }
        // double tmp_cx = phi_vertex[tri_a_index_loc[itmp]].x();
        // double tmp_cy = phi_vertex[tri_a_index_loc[itmp]].y();
        // double tmp_cz = phi_vertex[tri_a_index_loc[itmp]].z();

        // hash_cor(tmp_cx,tmp_cy,tmp_cz,hash_tmp);
        // auto hash_sear = ver_glo_index_map.find(hash_tmp);
        // if( hash_sear != ver_glo_index_map.end() ){
        //     tri_a_index_tmp.push_back( hash_sear->second );
        //     //printf("before:%.15e, %.15e,
        //     %.15e\n",tmp_cx,tmp_cy,tmp_cz);
        //     //printf("after:%.15e, %.15e,
        //     %.15e\n",vx_un_res[hash_sear->second],vy_un_res[hash_sear->second],vz_un_res[hash_sear->second]);
        //     //std::cout<<std::endl;
        // }
        // else{
        //     //std::cout << "rank: " << rankvalues << " " << "index: " <<
        //     tri_a_index_loc[itmp] << " " <<  tmp_cx << " " << tmp_cy << "
        //     " << tmp_cz << std::endl; std::cout << "Search in
        //     ver_glo_index_map not found. Must be something wrong." <<
        //     std::endl;
        // }
    }

    for (int itmp = 0; itmp < tri_b_index_loc.size(); itmp++) {
        auto it = ver_glo_index_map.find(phi_vertex[tri_b_index_loc[itmp]]);
        if (it != ver_glo_index_map.end()) {
            tri_b_index_tmp.push_back(it->second);
        }
        else {
            printf("Error: can not find the vertex in local processor");
        }
        // double tmp_cx = phi_vertex[tri_b_index_loc[itmp]].x();
        // double tmp_cy = phi_vertex[tri_b_index_loc[itmp]].y();
        // double tmp_cz = phi_vertex[tri_b_index_loc[itmp]].z();

        // hash_cor(tmp_cx,tmp_cy,tmp_cz,hash_tmp);
        // auto hash_sear = ver_glo_index_map.find(hash_tmp);
        // if( hash_sear != ver_glo_index_map.end() ){
        //     tri_b_index_tmp.push_back( hash_sear->second );
        //     //printf("before:%.15e, %.15e,
        //     %.15e\n",tmp_cx,tmp_cy,tmp_cz);
        //     //printf("after:%.15e, %.15e,
        //     %.15e\n",vx_un_res[hash_sear->second],vy_un_res[hash_sear->second],vz_un_res[hash_sear->second]);
        //     //std::cout<<std::endl;
        // }
        // else{
        //     //std::cout << "rank: " << rankvalues << " " << "index: " <<
        //     tri_b_index_loc[itmp] << " " <<  tmp_cx << " " << tmp_cy << "
        //     " << tmp_cz << std::endl; std::cout << "Search in
        //     ver_glo_index_map not found. Must be something wrong." <<
        //     std::endl;
        // }
    }

    for (int itmp = 0; itmp < tri_c_index_loc.size(); itmp++) {
        auto it = ver_glo_index_map.find(phi_vertex[tri_c_index_loc[itmp]]);
        if (it != ver_glo_index_map.end()) {
            tri_c_index_tmp.push_back(it->second);
        }
        else {
            printf("Error: can not find the vertex in local processor");
        }
        // double tmp_cx = phi_vertex[tri_c_index_loc[itmp]].x();
        // double tmp_cy = phi_vertex[tri_c_index_loc[itmp]].y();
        // double tmp_cz = phi_vertex[tri_c_index_loc[itmp]].z();

        // hash_cor(tmp_cx,tmp_cy,tmp_cz,hash_tmp);
        // auto hash_sear = ver_glo_index_map.find(hash_tmp);
        // if( hash_sear != ver_glo_index_map.end() ){
        //     tri_c_index_tmp.push_back( hash_sear->second );
        //     //printf("before:%.15e, %.15e,
        //     %.15e\n",tmp_cx,tmp_cy,tmp_cz);
        //     //printf("after:%.15e, %.15e,
        //     %.15e\n",vx_un_res[hash_sear->second],vy_un_res[hash_sear->second],vz_un_res[hash_sear->second]);
        //     //std::cout<<std::endl;
        // }
        // else{
        //     //std::cout << "rank: " << rankvalues << " " << "index: " <<
        //     tri_c_index_loc[itmp] << " " <<  tmp_cx << " " << tmp_cy << "
        //     " << tmp_cz << std::endl; std::cout << "Search in
        //     ver_glo_index_map not found. Must be something wrong." <<
        //     std::endl;
        // }
    }

    // MPI communication
    details::MPI_gather(MPI_COMM_WORLD, tri_a_index_tmp, c1_tmp);
    details::MPI_gather(MPI_COMM_WORLD, tri_b_index_tmp, c2_tmp);
    details::MPI_gather(MPI_COMM_WORLD, tri_c_index_tmp, c3_tmp);

    clean_triangle(c1_tmp, c2_tmp, c3_tmp, c1_res, c2_res, c3_res, vx_un_res,
                   vy_un_res, vz_un_res, vx_res, vy_res, vz_res);
}

void filter_point(std::vector<double> &vx_res, std::vector<double> &vy_res,
                  std::vector<double> &vz_res,
                  std::vector<double> &vertex_nor_x,
                  std::vector<double> &vertex_nor_y,
                  std::vector<double> &vertex_nor_z,
                  const std::vector<double> &vertex_nor_num,
                  const std::vector<double> &vertex_nor_area) {
    int num = vx_res.size();

    std::vector<double> vx_new;
    std::vector<double> vy_new;
    std::vector<double> vz_new;

    std::vector<double> nx_new;
    std::vector<double> ny_new;
    std::vector<double> nz_new;

    for (int i = 0; i < num; i++) {
        if (vertex_nor_num[i] > 2.5) {
            vx_new.push_back(vx_res[i]);
            vy_new.push_back(vy_res[i]);
            vz_new.push_back(vz_res[i]);

            nx_new.push_back(vertex_nor_x[i]);
            ny_new.push_back(vertex_nor_y[i]);
            nz_new.push_back(vertex_nor_z[i]);
        }
    }

    vx_res.swap(vx_new);
    vy_res.swap(vy_new);
    vz_res.swap(vz_new);

    vertex_nor_x.swap(nx_new);
    vertex_nor_y.swap(ny_new);
    vertex_nor_z.swap(nz_new);
}

// compute nodal normal vector
void compute_normal(
    const std::vector<double> &vx_res, const std::vector<double> &vy_res,
    const std::vector<double> &vz_res, const std::vector<int> &c1_res,
    const std::vector<int> &c2_res, const std::vector<int> &c3_res,
    std::vector<double> &vertex_nor_x, std::vector<double> &vertex_nor_y,
    std::vector<double> &vertex_nor_z, std::vector<double> &vertex_nor_num,
    std::vector<double> &vertex_nor_area) {
    // Resize vertex_nor size
    vertex_nor_x.assign(vx_res.size(), 0.0);
    vertex_nor_y.assign(vx_res.size(), 0.0);
    vertex_nor_z.assign(vx_res.size(), 0.0);
    vertex_nor_num.assign(vx_res.size(), 0.0);
    vertex_nor_area.assign(vx_res.size(), 0.0);

    int rankvalues = details::MPI_rank(MPI_COMM_WORLD);

    // Normal vector
    if (rankvalues == 0) {
        for (int ielem = 0; ielem < c1_res.size(); ielem++) {
            int c1_index = c1_res[ielem];
            int c2_index = c2_res[ielem];
            int c3_index = c3_res[ielem];

            // Get triangle vertex coordinate
            Point3D v1_tmp(vx_res[c1_index], vy_res[c1_index],
                           vz_res[c1_index]);
            Point3D v2_tmp(vx_res[c2_index], vy_res[c2_index],
                           vz_res[c2_index]);
            Point3D v3_tmp(vx_res[c3_index], vy_res[c3_index],
                           vz_res[c3_index]);

            // Calculate the normal vector for triangle
            double unit_scale = 1e6;
            Point3D e12 = (v2_tmp - v1_tmp) * unit_scale;
            Point3D e13 = (v3_tmp - v1_tmp) * unit_scale;
            Point3D nor = e12.cross(e13);

            double nor_mag = nor.norm();
            double tri_area = nor_mag / 2.0;
            nor = nor / (nor_mag + 1e-30);

            // Add normal vector to vertex vector
            vertex_nor_x[c1_index] += nor.x();
            vertex_nor_x[c2_index] += nor.x();
            vertex_nor_x[c3_index] += nor.x();

            vertex_nor_y[c1_index] += nor.y();
            vertex_nor_y[c2_index] += nor.y();
            vertex_nor_y[c3_index] += nor.y();

            vertex_nor_z[c1_index] += nor.z();
            vertex_nor_z[c2_index] += nor.z();
            vertex_nor_z[c3_index] += nor.z();

            // Count the number
            vertex_nor_num[c1_index] += 1.0;
            vertex_nor_num[c2_index] += 1.0;
            vertex_nor_num[c3_index] += 1.0;

            vertex_nor_area[c1_index] += tri_area;
            vertex_nor_area[c2_index] += tri_area;
            vertex_nor_area[c3_index] += tri_area;
        }

        // Average the normal vector
        for (int itmp = 0; itmp < vertex_nor_x.size(); itmp++) {
            if (vertex_nor_num[itmp] > 0) {
                vertex_nor_x[itmp] =
                    vertex_nor_x[itmp] / (vertex_nor_num[itmp] * 1.0);
                vertex_nor_y[itmp] =
                    vertex_nor_y[itmp] / (vertex_nor_num[itmp] * 1.0);
                vertex_nor_z[itmp] =
                    vertex_nor_z[itmp] / (vertex_nor_num[itmp] * 1.0);

                // normalize normal
                double mag_tol = 1e-30;
                double nor_mag =
                    std::sqrt(vertex_nor_x[itmp] * vertex_nor_x[itmp] +
                              vertex_nor_y[itmp] * vertex_nor_y[itmp] +
                              vertex_nor_z[itmp] * vertex_nor_z[itmp]) +
                    mag_tol;

                vertex_nor_x[itmp] = vertex_nor_x[itmp] / nor_mag;
                vertex_nor_y[itmp] = vertex_nor_y[itmp] / nor_mag;
                vertex_nor_z[itmp] = vertex_nor_z[itmp] / nor_mag;
            }
            else {
                std::cout << "Zero normal vectors find "
                             "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                          << std::endl;
            }
        }
    }

    details::MPI_broadcast(MPI_COMM_WORLD, vertex_nor_x);
    details::MPI_broadcast(MPI_COMM_WORLD, vertex_nor_y);
    details::MPI_broadcast(MPI_COMM_WORLD, vertex_nor_z);

    details::MPI_broadcast(MPI_COMM_WORLD, vertex_nor_num);
    details::MPI_broadcast(MPI_COMM_WORLD, vertex_nor_area);
}

} // namespace details

template <typename VolumeMesh, typename SurfaceMesh>
void get_free_surface(const VolumeMesh &mesh, const std::vector<double> &phid0,
                      SurfaceMesh &surface) {
    std::vector<Point3D> vertices;
    std::vector<std::vector<int>> elements;
    add_triangle(mesh, phid0, vertices, elements);
    surface.vertices = std::move(vertices);
    // flatten elements in to surface.elements
    surface.elements.reserve(elements.size() * 3);
    for (const auto &e : elements) {
        surface.elements.insert(surface.elements.end(), e.begin(), e.end());
    }
}

void compute_normal_vector(const TriangleMesh &mesh,
                           std::vector<Point3D> &vertices,
                           std::vector<Point3D> &normals) {
    std::vector<double> point_x(mesh.vertices.size());
    std::vector<double> point_y(mesh.vertices.size());
    std::vector<double> point_z(mesh.vertices.size());
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        point_x[i] = mesh.vertices[i].x();
        point_y[i] = mesh.vertices[i].y();
        point_z[i] = mesh.vertices[i].z();
    }
    std::vector<int> cell_index0(mesh.elements.size() / 3);
    std::vector<int> cell_index1(mesh.elements.size() / 3);
    std::vector<int> cell_index2(mesh.elements.size() / 3);
    for (std::size_t i = 0; i < mesh.elements.size() / 3; ++i) {
        cell_index0[i] = mesh.elements[i * 3 + 0];
        cell_index1[i] = mesh.elements[i * 3 + 1];
        cell_index2[i] = mesh.elements[i * 3 + 2];
    }
    std::vector<double> vertex_nor_x(mesh.vertices.size());
    std::vector<double> vertex_nor_y(mesh.vertices.size());
    std::vector<double> vertex_nor_z(mesh.vertices.size());
    std::vector<double> vertex_nor_num(mesh.vertices.size());
    std::vector<double> vertex_nor_area(mesh.vertices.size());
    std::vector<double> degrees;
    std::vector<double> areas;
    compute_normal(point_x, point_y, point_z, cell_index0, cell_index1,
                   cell_index2, vertex_nor_x, vertex_nor_y, vertex_nor_z,
                   degrees, areas);

    filter_point(point_x, point_y, point_z, vertex_nor_x, vertex_nor_y,
                 vertex_nor_z, degrees, areas);

    vertices.clear();
    vertices.reserve(point_x.size());
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        vertices.emplace_back(point_x[i], point_y[i], point_z[i]);
    }
    normals.clear();
    normals.reserve(vertex_nor_x.size());
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        normals.emplace_back(vertex_nor_x[i], vertex_nor_y[i], vertex_nor_z[i]);
    }
}

template <typename = void> struct Redistance {};

template <> struct Redistance<TetrahedronMesh> {
    using VolumeMesh = TetrahedronMesh;
    using SurfaceMesh = TriangleMesh;

    VolumeMesh domain;
    SurfaceMesh free_surface_mesh;
    Graph global_vertex_connectivity;
    std::vector<double> scalar_field;
    std::vector<double> phi;

    Redistance(const VolumeMesh &mesh) : domain(mesh) {
        get_vertex_connectivity_in_global_patch(mesh,
                                                global_vertex_connectivity);
    }

    //
    void init(const std::vector<double> &phid0,
              int small_droplet_tolerance = 100) {
        // Gather the levelset function
        details::MPI_DataDistribution dist;
        std::vector<int> gid;
        for (int ivtx = 0; ivtx < domain.vertices.size(); ivtx++) {
            gid.push_back(mesh.vertex_local2global[ivtx]);
        }
        phi = phid0;
        dist.gather(gid, phi);
        scalar_field = dist.values;
        // flip the sign of the levelset function in small droplets
        overwrite_small_droplets(small_droplet_tolerance);
        dist.scatter(gid, phi);

        get_free_surface(domain, phi, isosurface);
    }

    // Compute levelset sign distance function here
    void compute_levelset_distance(const VolumeMesh &mesh,
                                   std::vector<double> &phid0,
                                   const std::vector<int> &phi_factor,
                                   const std::vector<std::size_t> &d2v_map,
                                   const std::vector<double> &vx_res,
                                   const std::vector<double> &vy_res,
                                   const std::vector<double> &vz_res,
                                   const std::vector<double> &vertex_nor_x,
                                   const std::vector<double> &vertex_nor_y,
                                   const std::vector<double> &vertex_nor_z) {
        std::vector<Point3D> phi_inter_glo;

        // Start to calculate the distance between point and point cloud
        // (bounding box) Copy to point vector
        for (int iphi = 0; iphi < vx_res.size(); iphi++) {
            phi_inter_glo.push_back(
                Point(vx_res[iphi], vy_res[iphi], vz_res[iphi]));
        }

        bool ismaster = details::MPI_rank(MPI_COMM_WORLD) == 0;
        if (ismaster and phi_inter_glo.empty()) {
            std::cout
                << "The triangulization of phi=0 has zero point. "
                << "Please check the threshold in bubble dropping process.";
        }

        // Get the bounding box for point cloud
        OctreePoint phi0_Bound_Box_Tree;
        phi0_Bound_Box_Tree.set_box(phi_inter_glo);
        phi0_Bound_Box_Tree.insert_point(phi_inter_glo);

        // Get the value of phi in each processor
        // std::vector<double> phid0_vec;
        // phid0->vector()->get_local(phid0_vec);
        auto &phid0_vec = phid0;

        double phi_sign;
        for (int iphi = 0; iphi < phid0_vec.size(); iphi++) {
            // Get the vertex index from dof index
            std::size_t phi_ind = d2v_map[iphi];

            // Get the coordinate of corresponding point
            // Point3D
            // tmp_point(local_cor[3*phi_ind],local_cor[3*phi_ind+1],local_cor[3*phi_ind+2]);
            const auto &tmp_point = mesh.vertex[phi_ind];

            // Use bounding box to get the closest distance from point to point
            // cloud
            std::pair<int, double> dist_pair;
            int oct_tree_index;
            double oct_tree_dist;
            phi0_Bound_Box_Tree.search_point(tmp_point, oct_tree_index,
                                             oct_tree_dist);

            dist_pair = std::make_pair(oct_tree_index, oct_tree_dist);

            // Get the nearest point in the cloud
            Point3D near_point_cloud = phi_inter_glo[dist_pair.first];

            // Calculate the averaged normal on vertex
            Point3D nor_vertex_cal{vertex_nor_x[dist_pair.first],
                                   vertex_nor_y[dist_pair.first],
                                   vertex_nor_z[dist_pair.first]};

            // Get the line connecting vertex point and near point in cloud.

            // Calculate projection length over vertex normal direction
            double dist_proj =
                std::abs(nor_vertex_cal.dot(tmp_point - near_point_cloud));

            /*
            if( near_edge.dot(nor_vertex_cal) > 0.0 ){
                phi_sign =  1.0;
            }
            else{
                phi_sign = -1.0;
            }
            */

            if (phid0_vec[iphi] > 0.0) {
                phi_sign = 1.0;
            }
            else {
                phi_sign = -1.0;
            }

            // Change dropped point sign
            Vertex phi_ver = Vertex(*mesh, phi_ind);
            double phi_factor_val = phi_factor[phi_ver.global_index()] * 1.0;
            phi_sign = phi_sign * phi_factor_val;

            double phi_mag = std::abs(phid0_vec[iphi]);
            double phi_tol = 1.5 * epslen_;

            if (phi_mag > phi_tol) {
                // Use distance to point cloud
                phid0_vec[iphi] = phi_sign * dist_pair.second;
            }
            else {
                // Use projected distance
                phid0_vec[iphi] = phi_sign * dist_proj;
            }

            /*
            // correction for coarse mesh
            if( std::abs( tmp_point.y() ) > 1.0e-4 ){
                            if( std::abs( tmp_point.z() ) < 4e-5 ){
                                    phid0_vec[iphi] = 0.0 - tmp_point.z();
                            }
                    }
            */
        }
    }

    // Compute edge connectivity
    void compute_connect(std::vector<std::set<int>> &node_connect_glo,
                         std::shared_ptr<Mesh> &mesh) {

        // Get some some mesh
        const std::size_t tdim = mesh->topology().dim();
        const std::size_t gdim = mesh->geometry().dim();
        const std::size_t num_local_vertices = mesh->num_entities(0);
        const std::size_t num_global_vertices = mesh->num_entities_global(0);

        std::vector<int> node_connect_loc_a;
        std::vector<int> node_connect_loc_b;
        std::vector<int> node_connect_glo_a;
        std::vector<int> node_connect_glo_b;
        std::vector<std::set<int>> node_connect_loc(num_global_vertices,
                                                    std::set<int>());

        node_connect_glo.resize(num_global_vertices);
        int rankvalues = details::MPI_rank(MPI_COMM_WORLD);

        // Compute connectivity
        int cell_ind = 0;
        std::vector<int> cell_vec(4, 0);
        for (CellIterator cell(*mesh); !cell.end(); ++cell) {
            int vertex_ind = 0;
            for (VertexIterator v(*cell); !v.end(); ++v) {
                int global_ind = v->global_index();
                cell_vec[vertex_ind] = global_ind;
                vertex_ind++;
            }

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    if (i == j)
                        continue;

                    node_connect_loc[cell_vec[i]].insert(cell_vec[j]);
                    node_connect_loc[cell_vec[j]].insert(cell_vec[i]);
                }
            }
            cell_ind++;
        }

        std::set<int>::iterator node_it;
        for (int i = 0; i < num_global_vertices; i++) {
            for (node_it = node_connect_loc[i].begin();
                 node_it != node_connect_loc[i].end(); ++node_it) {
                int val = *node_it;
                node_connect_loc_a.push_back(i);
                node_connect_loc_b.push_back(val);
            }
        }

        details::MPI_gather(MPI_COMM_WORLD, node_connect_loc_a,
                            node_connect_glo_a);
        details::MPI_gather(MPI_COMM_WORLD, node_connect_loc_b,
                            node_connect_glo_b);

        if (rankvalues == 0) {
            for (int i = 0; i < node_connect_glo_a.size(); i++) {
                node_connect_glo[node_connect_glo_a[i]].insert(
                    node_connect_glo_b[i]);
            }
        }
    }

    // void redistance_prep(std::shared_ptr<Mesh> &mesh) {
    //  tec_map_compute(this->tec_comm_local_ind_full,this->tec_comm_local_ind_reduce,this->tec_comm_global_ind_reduce_loc_glo,mesh);
    //  compute_connect(this->elem_connect, mesh);
    // }

    void overwrite_small_droplets(int num_tol) {

        int rankvalues = details::MPI_rank(MPI_COMM_WORLD);
        if (rankvalues) {
            return;
        }

        std::vector<std::vector<std::size_t>> components;
        connected_components(this->global_vertex_connectivity, components,
                             [&](std::size_t i, std::size_t j) {
                                 return scalar_field[i] * scalar_field[j] > 0.0;
                             });
        for (int i = 0; i < color_num.size(); i++) {
            std::cout << "    Volume color " << i << " number: " << color_num[i]
                      << std::endl;
        }

        for (const auto &patch : components) {
            if (patch.size() < num_tol)
                continue;
            std::for_each(patch.begin(), patch.end(),
                          [&](std::size_t i) { scalar_field[i] *= -1; });
        }
    }

    // void phi_sign_init(std::vector<int> &phi_sign,
    //                    std::shared_ptr<Mesh> &mesh) {
    //     const std::size_t num_global_vertices = mesh->num_entities_global(0);

    //     phi_sign.resize(num_global_vertices);
    //     std::fill(phi_sign.begin(), phi_sign.end(), 1);
    // }

    void geo_redistance(std::shared_ptr<Mesh> &mesh,
                        std::shared_ptr<Function> phid0,
                        std::vector<std::size_t> &d2v_map,
                        std::vector<dolfin::la_index> &v2d_map) {
        // debug use
        /*
        {
            File phifile("comp/phi_ini.pvd");
            auto phiout = *phid0;
            phifile << phiout;
        }
        */

        // std::vector<Point3D>             phi_vertex;
        // std::vector<std::vector<int>>  phi_connect;
        auto &phi_vertex = isosurface.vertex;
        auto &phi_connect = isosurface.connectivity;

        std::vector<double> vx_res;
        std::vector<double> vy_res;
        std::vector<double> vz_res;

        std::vector<int> c1_res;
        std::vector<int> c2_res;
        std::vector<int> c3_res;

        std::vector<double> vertex_nor_x;
        std::vector<double> vertex_nor_y;
        std::vector<double> vertex_nor_z;

        std::vector<double> vertex_nor_num;
        std::vector<double> vertex_nor_area;

        std::vector<int> phi_sign;

        bool ismaster = details::MPI_rank(MPI_COMM_WORLD) == 0;

        // Triangulation of the phi=0
        // add_triangle(mesh, phid0, phi_vertex, phi_connect);

        // if(ismaster) info("Start to combine triangle.");
        //  Combine triangle from different processor and remove duplicate
        combine_triangle(phi_vertex, phi_connect, vx_res, vy_res, vz_res,
                         c1_res, c2_res, c3_res);

        // if(ismaster) info("Start to compute normal.");
        //  Compute normal vector at vertex
        // compute_normal(vx_res, vy_res, vz_res, c1_res, c2_res, c3_res,
        //                vertex_nor_x, vertex_nor_y, vertex_nor_z,
        //                vertex_nor_num, vertex_nor_area);
        std::vector<Point3D> vertices;
        std::vector<Point3D> normal_vectors;
        compute_normal_vector(isosurface, vertices, normal_vectors);
        // Laplace smoothing
        // laplace_smooth(vx_res, vy_res, vz_res, c1_res, c2_res, c3_res);

        // debug use
        // write_triangulation(vx_res, vy_res, vz_res,\
        //             vertex_nor_x, vertex_nor_y, vertex_nor_z,\
        //             c1_res, c2_res, c3_res );

        // filter_point(vx_res, vy_res, vz_res, vertex_nor_x, vertex_nor_y,
        //              vertex_nor_z, vertex_nor_num, vertex_nor_area);

        // The volume mesh is divided by the phi=0 surface. Traverse the
        // vertices of the mesh and color the vertices on the two sides of the
        // surface.

        // if(ismaster) info("Start to compute levelset distance.");
        //  Levelset redistancing calculation
        compute_levelset_distance(mesh, phid0, phi_sign, d2v_map, vx_res,
                                  vy_res, vz_res, vertex_nor_x, vertex_nor_y,
                                  vertex_nor_z);

        // debug use
        /*
        {
            File phifile("comp/phi_redis.pvd");
            auto phiout = *phid0;
            phifile << phiout;
        }
        */
    }
};
} // namespace GeoRd
#endif // __REDIS_H__