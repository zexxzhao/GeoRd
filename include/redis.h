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
                  std::vector<int> &phi_connect) {
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
            // phi_connect.push_back(tri_con_tmp);
            phi_connect.insert(phi_connect.end(), tri_con_tmp.begin(),
                               tri_con_tmp.end());

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
                // phi_connect.push_back(tri_con_tmp);
                phi_connect.insert(phi_connect.end(), tri_con_tmp.begin(),
                                   tri_con_tmp.end());
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

// clean the triangles
template <typename T, std::enable_if<std::is_integral<T>::value, int>::type = 0>
void clean_triangles(std::vector<Point3D> &vertices, std::vector<T> &elements) {

    // remove duplicate vertices
    details::UnorderedMap<Point3D, std::size_t> vertex_map;
    std::size_t vertex_counter = 0;
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        auto it = vertex_map.find(vertices[i]);
        if (it != vertex_map.end()) {
            continue;
        }
        vertex_map[vertices[i]] = vertex_counter;
        ++vertex_counter;
    }

    // remove duplicate triangles
    using Triangle = details::Triple<T>;
    details::UnorderedSet<Triangle> triangle_set;
    for (std::size_t i = 0; i < elements.size(); i += 3) {
        Triangle triangle(elements[i], elements[i + 1], elements[i + 2]);
        auto it = triangle_set.find(triangle);
        if (it != triangle_set.end()) {
            continue;
        }
        triangle_set.insert(triangle);
    }

    // update triangle indices following the new vertex indices
    std::for_each(triangle_set.begin(), triangle.end(), [&](Triangle &t) {
        t[0] = vertex_map[vertices[t[0]]];
        t[1] = vertex_map[vertices[t[1]]];
        t[2] = vertex_map[vertices[t[2]]];
    });

    // update vertices
    vertices.resize(vertex_map.size());
    for (auto &kv : vertex_map) {
        vertices[kv.second] = kv.first;
    }
    // update triangles
    elements.resize(triangle_set.size() * 3);
    std::size_t triangle_counter = 0;
    for (auto &t : triangle_set) {
        elements[triangle_counter * 3 + 0] = t[0];
        elements[triangle_counter * 3 + 1] = t[1];
        elements[triangle_counter * 3 + 2] = t[2];
        ++triangle_counter;
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

template <typename VolumeMesh, typename SurfaceMesh>
void get_free_surface(const VolumeMesh &mesh, const std::vector<double> &phid0,
                      SurfaceMesh &surface) {
    std::vector<Point3D> vertices;
    std::vector<int> elements;
    add_triangle(mesh, phid0, vertices, elements);
    // Gather vertices and elements
    std::vector<Point3D> vertices_all;
    std::vector<int> elements_all;
    details::MPI_allgather(MPI_COMM_WORLD, vertices, vertices_all);
    details::MPI_allgather(MPI_COMM_WORLD, elements, elements_all);
    // Clean triangles
    details::clean_triangles(vertices_all, elements_all);
    // Place the result into the surface mesh
    surface.vertices = std::move(vertices_all);
    surface.elements = std::move(elements_all);
}

} // namespace details

template <typename = void> struct Redistance {};

template <> struct Redistance<TetrahedronMesh> {
    using VolumeMesh = TetrahedronMesh;
    using SurfaceMesh = TriangleMesh;

    VolumeMesh domain;
    struct Representation {
        std::vector<Point3D> vertices;
        std::vector<Point3D> normals;
    };
    Representation representation;
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
        if (details::MPI_rank() == 0) {
            std::vector<std::vector<std::size_t>> volume_components;
            overwrite_small_droplets(small_droplet_tolerance,
                                     volume_components);
            // TODO: assign signs to scalar_field
        }
        dist.scatter(gid, phi);

        // Generate the free surface mesh
        SurfaceMesh free_surface_mesh;
        get_free_surface(domain, phi, free_surface_mesh);
        // Compute the normal vector of the free surface
        details::compute_normal_vector(
            free_surface_mesh, representation.vertices, representation.normals);
    }

    // Assign the level set function to the vertices of the mesh
    void redistance(std::vector<double> &phid, double eps) const {

        // Ensure positive eps
        eps = std::abs(eps);

        // Build the octree
        Octree<Point3D> tree;
        tree.set_box(representation.vertices);
        tree.insert_point(representation.vertices);

        // Compute the distance between the vertices and the free surface
        for (int ivtx = 0; ivtx < phid.size(); ++ivtx) {
            auto p = domain.vertices[ivtx];
            int idx_nearest;
            double distance_nearest;
            tree.search_point(p, idx_nearest, distance_nearest);
            if (std::abs(phid[ivtx]) < eps) {
                const auto p_nearest = representation.vertices[idx_nearest];
                const auto n_nearest = representation.normals[idx_nearest];
                distance_nearest = std::abs((p - p_nearest).dot(n_nearest));
            }
            if (phid[ivtx] < 0) {
                distance_nearest = -distance_nearest;
            }
            phid[ivtx] = distance_nearest;
        }
    }

    void overwrite_small_droplets(
        int num_tol, std::vector<std::vector<std::size_t>> &components) {

        int rankvalues = details::MPI_rank(MPI_COMM_WORLD);
        if (rankvalues) {
            return;
        }

        connected_components(this->global_vertex_connectivity, components,
                             [&](std::size_t i, std::size_t j) {
                                 return scalar_field[i] * scalar_field[j] > 0.0;
                             });
        for (int i = 0; i < color_num.size(); i++) {
            std::cout << "    Volume color " << i << " number: " << color_num[i]
                      << std::endl;
        }
        // get inter-patch connectivity
        std::vector<std::vector<std::size_t>> inter_patch_connectivity;
        get_inter_patch_connectivity(this->global_vertex_connectivity,
                                     inter_patch_connectivity);

        // attach the small droplets to the large neighboring droplets
        // 1. find the small droplets ID
        std::vector<std::size_t> small_droplets_id;
        for (std::size_t i = 0; i < components.size(); ++i) {
            if (components[i].size() < num_tol) {
                small_droplets_id.push_back(i);
            }
        }
        // 2. find the corresponding large neighboring droplets ID
        std::vector<std::size_t> large_droplets_id(small_droplets_id.size());
        std::vector<bool> found(small_droplets_id.size(), false);
        do {
            for (std::size_t i = 0; i < small_droplets_id.size(); ++i) {
                if (found[i]) {
                    continue;
                }
                // find the large neighboring droplets ID
                const auto neighbors =
                    inter_patch_connectivity[small_droplets_id[i]];
                for (const auto &neighbor : neighbors) {
                    if (components[neighbor].size() >= num_tol) {
                        large_droplets_id[i] = neighbor;
                        found[i] = true;
                        break;
                    }
                    if (found[neighbor]) {
                        large_droplets_id[i] = large_droplets_id[neighbor];
                        found[i] = true;
                        break;
                    }
                }
            }
        } while (std::find(found.begin(), found.end(), false) != found.end());

        // merge the small droplets into the large neighboring droplets
        for (std::size_t i = 0; i < small_droplets_id.size(); ++i) {
            auto &small_droplet = components[small_droplets_id[i]];
            auto &large_droplet = components[large_droplets_id[i]];
            large_droplet.insert(large_droplet.end(), small_droplet.begin(),
                                 small_droplet.end());
            small_droplet.clear();
        }
        // erase the empty components
        components.erase(std::remove_if(components.begin(), components.end(),
                                        [](const std::vector<std::size_t> &c) {
                                            return c.empty();
                                        }),
                         components.end());
    }

    //
    void get_inter_patch_connectivity(
        const Graph &vertex_connectivity,
        const std::vector<std::vector<std::size_t>> &components,
        std::vector<std::vector<std::size_t>> &inter_patch_connectivity) const {
        inter_patch_connectivity.clear();
        inter_patch_connectivity.resize(components.size());
        // find the component with the largest number of vertices as the
        // starting point
        auto max_component =
            std::max_element(components.begin(), components.end(),
                             [](const std::vector<std::size_t> &a,
                                const std::vector<std::size_t> &b) {
                                 return a.size() < b.size();
                             });
        // find a representative vertex in each component
        std::vector<std::size_t> representative_vertex(components.size());
        for (std::size_t i = 0; i < components.size(); ++i) {
            representative_vertex[i] = components[i][0];
        }
        // Use greedy to identity the connectivity between components
        for (int i = 0; i < representative_vertex.size(); ++i) {
            for (int j = i + 1; j < representative_vertex.size(); ++j) {
                std::vector<std::size_t> path;
                details::get_path(vertex_connectivity, representative_vertex[i],
                                  representative_vertex[j], path);
                if (path.size() > 0) {
                    inter_patch_connectivity[i].push_back(j);
                    inter_patch_connectivity[j].push_back(i);
                }
            }
        }
    }
};
} // namespace GeoRd
#endif // __REDIS_H__