#ifndef __REDIS_H__
#define __REDIS_H__
#include <set>
#include <unorder_map>
#include "Octree.h"
namespace GeoRd {

namespace details {

template <typename T>
double cos(const T& v1, const T& v2) {
    return v1.dot(v2) / (v1.norm() * v2.norm() + 1e-30);
}

inline std::string d2str(double x){
    const double scale   = 1.0e6;
    double x_scale = (x + 1e2) * scale;
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(8) << x_scale;
    return ss.str();
}

inline void hash_cor(double x, double y, double z, std::string& res_str){
    std::string x_str,y_str,z_str;

    res_str.clear();
    x_str = d2str(x);
    y_str = d2str(y);
    z_str = d2str(z);

    res_str = x_str + "," + y_str + "," + z_str;
};

inline void hash_con(int x, int y, int z, std::string& res_str){
    std::string x_str,y_str,z_str;

    int r1,r2,r3;
    r1 = std::min( std::min(x,y), z);
    r3 = std::max( std::max(x,y), z);
    r2 = x+y+z - r1 - r3;

    res_str.clear();
    x_str = std::to_string(r1);
    y_str = std::to_string(r2);
    z_str = std::to_string(r3);

    res_str = x_str + "," + y_str + "," + z_str;
};

template <typename T> using Triplet = std::array<T, 3>;

template <typename T, typename = void>
struct HashTable {
    std::size_t operator()(const T& t) const {
        return std::hash<T>()(t);
    }
};

template <typename T>
struct HashTable<Triplet<T>, typename std::enable_if<std::is_floating_point<T>::value>::type> {
    std::size_t operator()(const Triplet<T>& t) const {
        std::string str;
        hash_cor(std::get<0>(t), std::get<1>(t), std::get<2>(t), str);
        return std::hash<std::string>()(str);
    }
};

template <typename T>
struct HashTable<Triplet<T>, typename std::enable_if<std::is_integral<T>::value>::type> {
    std::size_t operator()(const Triplet<T>& t) const {
        std::string str;
        hash_con(std::get<0>(t), std::get<1>(t), std::get<2>(t), str);
        return std::hash<std::string>()(str);
    }
};

template <>
struct HashTable<Point3D> {
    std::size_t operator()(const Point3D& p) const {
        return HashTable<Triplet<double>>()(Triplet<double>{p[0], p[1], p[2]});
    }
};

// Primary template for KeyEqual<T> using SFINAE
template <typename T, typename = void>
struct KeyEqual {
    bool operator()(const T& t1, const T& t2) const {
        return t1 == t2;
    }
};


// Partial specialization for KeyEqual<Triplet<T>> when T is floating point type using SFINAE
template <typename T>
struct KeyEqual<Triplet<T>, typename std::enable_if<std::is_floating_point<T>::value>::type> {
    bool operator()(const Triplet<T>& t1, const Triplet<T>& t2) const {
        return std::abs(std::get<0>(t1) - std::get<0>(t2)) < 1e-6 &&
               std::abs(std::get<1>(t1) - std::get<1>(t2)) < 1e-6 &&
               std::abs(std::get<2>(t1) - std::get<2>(t2)) < 1e-6;
    }
};

// Partial specialization for KeyEqual<Triplet<T>> when T is integer type using SFINAE
template <typename T>
struct KeyEqual<Triplet<T>, typename std::enable_if<std::is_integral<T>::value>::type> {
    bool operator()(const Triplet<T>& t1, const Triplet<T>& t2) const {
        return std::get<0>(t1) == std::get<0>(t2) &&
               std::get<1>(t1) == std::get<1>(t2) &&
               std::get<2>(t1) == std::get<2>(t2);
    }
};

// Partial specialization for KeyEqual<Point3D> using SFINAE
template <>
struct KeyEqual<Point3D> {
    bool operator()(const Point3D& p1, const Point3D& p2) const {
        return KeyEqual<Triplet<double>>()(Triplet<double>{p1[0], p1[1], p1[2]},
                                           Triplet<double>{p2[0], p2[1], p2[2]});
    }
};

template<typename Key, typename Value>
using UnorderedMap = std::unordered_map<Key, Value, HashTable<Key>, KeyEqual<Key>>;

template <typename T> using Point3DMap = UnorderedMap<Point3D, T>;

} // namespace details


struct TetrahedronMesh {
    // Number of vertices per tetrahedron
    static constexpr std::size_t cell_vertex_count = 4;
    std::vector<Point3D> vertex;
    std::vector<std::array<std::size_t, cell_vertex_count>> connectivity;

    void get_cell_coordinates(std::size_t cell_id,
                              std::array<Point3D, cell_vertex_count>& cell_coordinates) const {
        for (std::size_t i = 0; i < cell_vertex_count; ++i) {
            cell_coordinates[i] = vertex[connectivity[cell_id][i]];
        }
    }
};

struct TriangleMesh {
    // number of vertices per triangle
    static constexpr std::size_t cell_vertex_count = 3;
    std::vector<Point3D> vertex;
    std::vector<std::array<std::size_t, cell_vertex_count>> connectivity;

    void get_cell_coordinates(std::size_t cell_id,
                              std::array<Point, cell_vertex_count>& cell_coordinates) const {
        for (std::size_t i = 0; i < cell_vertex_count; ++i) {
            cell_coordinates[i] = vertex[connectivity[cell_id][i]];
        }
    }
};

struct Redistance {
    TriangleMesh isosurface;
    std::vector<std::set<int>> elem_connect;
    // std::vector<la_index>      tec_comm_local_ind_full;
    // std::vector<la_index>      tec_comm_local_ind_reduce;
    // std::vector<int>           tec_comm_global_ind_reduce_loc_glo;

    int vol_color_flag = 1;
    int step_num;

    //* Coloring triangulation mesh using non-recursive DFS
    //* [in]  con: connectivity list
    //* [out] color: color list
    //
    void color_help(std::vector<std::set<int>> & con, \
            std::vector<int>           & color,\
            int   pos,\
            int & color_num,\
            int   color_val) {
        std::set<int>::iterator node_it;
        std::stack<int> stack;
        stack.push(pos);

        while( !stack.empty() ){
            int edge_start = stack.top();
            stack.pop();

            color_num++;
            color[edge_start] = color_val;
            
            for (node_it = con[edge_start].begin(); node_it != con[edge_start].end(); ++node_it) {
                int  edge_end  = *node_it;

                if( color[edge_end] == 0 ){
                    color[edge_end] = color_val;
                    stack.push( edge_end );
                }
            }
        }
    }

    // Coloring surface mesh
    // used to drop surface points
    void cal_color(std::vector<std::set<int>> & con, \
                   std::vector<int> & color,\
                   std::vector<int> & color_nlist) {
        int num = con.size();
        int color_val  = 1;
        int color_num  = 0;

        color.resize(num);
        std::fill(color.begin(),color.end(),0);
        
        for(int i=0;i<num;i++){
            if( color[i] == 0 ){
                color_help(con,color,i,color_num,color_val);
                if(color_num != 0 ){
                    color_val++;
                    color_nlist.push_back(color_num);
                }
            }
            color_num = 0;
        }
    };

    // DFS with non-recursive algorithm and scalar function phi
    void color_help(std::vector<std::set<int>> & con,
                    std::vector<double>        & phi,
            std::vector<int>           & color,
            int   pos,
            int & color_num,
            int   color_val) {
        std::set<int>::iterator node_it;
        std::stack<int> stack;
        stack.push(pos);

        while( !stack.empty() ){
            int edge_start = stack.top();
            stack.pop();

            color_num++;
            color[edge_start] = color_val;
            
            for (node_it = con[edge_start].begin(); node_it != con[edge_start].end(); ++node_it) {
                int  edge_end  = *node_it;
                bool same_sign = (phi[edge_start]*phi[edge_end] > 0.0);

                if( color[edge_end] == 0 and same_sign == true ){
                    color[edge_end] = color_val;
                    stack.push( edge_end );
                }
            }
        }
    };

    // Coloring surface mesh
    // used to change sign of phi related with dropped surface mesh
    void cal_color(	std::vector<std::set<int>> & con,  \
            std::vector<double>        & phi,  \
            std::vector<int>           & color,\
            std::vector<int>           & color_nlist)
    {
        int num = con.size();
        int color_val  = 1;
        int color_num  = 0;

        color.resize(num);
        std::fill(color.begin(),color.end(),0);
        
        for(int i=0;i<num;i++){
            if( color[i] == 0 ){
                color_help(con,phi,color,i,color_num,color_val);		
                if(color_num != 0 ){
                    color_val++;
                    color_nlist.push_back(color_num);
                }
            }
            color_num = 0;
        }
    };

    /**
     * @brief Make sure the normal vector of the triangle is pointing where phi is positive
     * 
     * @param[in] phi_vertex vertex coordinate of the triangulation mesh
     * @param[in] phi_tet_vertex vertex coordinate of the tetrahedral
     * @param[out] tri_con_tmp connectivity list of the triangle
     * @param[in] phival scalar function value of the tetrahedral mesh
     */
    void compute_triangle_winding_in_tetrahedron(
        const std::vector<Point3D> &phi_vertex,
        const std::vector<Point3D> &phi_tet_vertex,
        std::vector<int> &tri_con_tmp,
        const std::array<double, typename TetrahedronMesh::cell_vertex_count> &phival) {

        // Get triangle vertex coordinate
        auto v1_tmp = phi_vertex[tri_con_tmp[0]];
         v2_tmp = phi_vertex[tri_con_tmp[1]];
         v3_tmp = phi_vertex[tri_con_tmp[2]];

        // Calculate the normal vector for triangle
        const double unit_scale = 1e6;
        auto e12 = (v2_tmp - v1_tmp)*unit_scale;
        auto e13 = (v3_tmp - v1_tmp)*unit_scale;
        auto nor = e12.cross(e13);
        nor /= nor.norm() + 1e-30;

        // Check the direction of normal
        // get the index of the vertex with largest absolute phi value
        auto it = std::max_element(phival.begin(), phival.end(),
                                   [](double a, double b) { return std::abs(a) < std::abs(b);});
        auto sign = (*it > 0 ? 1 : -1);
        auto index = std::distance(phival.begin(), it);

        auto dot_val = (phi_tet_vertex[index] - v1_tmp).dot(nor) * sign;

        if(dot_val < 0.0){
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

        if(cos12 <= cos13 and cos12 <= cos23){
            diag_dir = 3;
        }
        if(cos13 <= cos12 and cos13 <= cos23){
            diag_dir = 2;
        }
        if(cos23 <= cos12 and cos23 <= cos13){
            diag_dir = 1;
        }

        return diag_dir;
    }

    void add_triangle(const TetrahedronMesh &mesh,
                      const std::vector<double> &phid0,
                      TriangleMesh &isosurface) {
        add_triangle(mesh, phid0, isosurface.vertex, isosurface.connectivity);
    }

    // Adding triangle
    // 1 triangle or 2 triangles
    void add_triangle(const TetrahedronMesh &mesh,
                      const std::vector<double> &phid0,
                      std::vector<Point3D> &phi_vertex,
                      std::vector<std::vector<int>>& phi_connect) {
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

        std::string hash_tmp;

        int ver_loc_index_cur = 0;
        //std::unordered_map<std::string, int> vertex_to_local_index;
        details::Point3DMap<std::size_t> vertex_to_local_index;

        // for (CellIterator cell(*mesh); !cell.end(); ++cell)
        for(int icell = 0; icell < mesh.cells.size(); ++icell) {
            // clear vector
            // std::vector<Point>().swap(phi_tmp_vertex);
            // std::vector<Point>().swap(phi_tet_vertex);
            // std::vector<int>().swap(phi_pos);
            // std::vector<int>().swap(phi_neg);
            std::vector<Point3D> phi_tmp_vertex;
            std::array<Point3D, typename Tetrahedron::cell_vertex_count> phi_tet_vertex;
            std::vector<int> phi_pos, phi_neg;

            // Get the coordinate of four vertex
            // mesh.get_cell_coordinates(icell, coordinate_dofs);
            mesh.get_cell_vertices(icell, phi_tet_vertex);
            // Get the phi value of four vertex
            std::array<double, typename Tetrahedron::cell_vertex_count> phival;
            for(int ivtx = 0; ivtx < typename TetrahedronMesh::cell_vertex_count; ivtx++) {
                phival[ivtx] = phid0[mesh.cells[icell][ivtx]];
                if(phival[ivtx] > 0.0) {
                    phi_pos.push_back(ivtx);
                } else {
                    phi_neg.push_back(ivtx);
                }
            }

            // save phi = 0 vertex (3 or 4 points)
            for(int ipos = 0; ipos < phi_pos.size(); ipos++) {
                for(int ineg = 0; ineg < phi_neg.size(); ineg++) {
                    // Use linear interpolation to get the phi=0 coordinate
                    auto ratio = (0.0 - phival[phi_neg[ineg]]) /
                                (phival[phi_pos[ipos]] - phival[phi_neg[ineg]] + 1e-30);
                    auto phi_coord = (1.0 - ratio) * phi_tet_vertex[phi_neg[ineg]] +
                                ratio * phi_tet_vertex[phi_pos[ipos]];

                    // Confusion: Why need to clip the data?
                    // Number near zero will cause problem in hash function
                    // Update: New version of hash function does not need to clip the data
                    //clip the data
                    double clip_tol = 1e-10;
                    for(int ic=0;ic<3;ic++){
                        if( std::abs(phi_cor[ic]) < clip_tol ){
                            phi_cor[ic] = 0.0;
                        }
                    }

                    // save phi=0 coordinate
                    phi_tmp_vertex.push_back(phi_coord);
                }
            }

            // 3 points case (Need to add one triangle)
            if(phi_tmp_vertex.size() == 3) {
                assert(phi_pos.size() == 1 or phi_neg.size() == 1);
                // Make sure it is empty
                std::vector<int> tri_con_tmp;

                for(int itmp=0;itmp < phi_tmp_vertex.size() ; itmp++) {

                    // if the vertex is already in the vector, just push back the index
                    // else, push back the vertex and the index
                    if (vertex_to_local_index.find(phi_tmp_vertex[itmp]) != vertex_to_local_index.end()) {
                        tri_con_tmp.push_back(vertex_to_local_index[phi_tmp_vertex[itmp]]);
                    } else {
                        phi_vertex.push_back(phi_tmp_vertex[itmp]);
                        tri_con_tmp.push_back(phi_vertex.size() - 1);
                        vertex_to_local_index[phi_tmp_vertex[itmp]] = ver_loc_index_cur;
                        ver_loc_index_cur++;
                    }
                }

                // if the triangle is degenerated, skip it
                if(tri_con_tmp[0]==tri_con_tmp[1] or
                   tri_con_tmp[0]==tri_con_tmp[2] or
                   tri_con_tmp[1]==tri_con_tmp[2] ) {
                    continue;
                }

                // Fix winding problem
                compute_triangle_winding_in_tetrahedron(phi_vertex, phi_tet_vertex, tri_con_tmp, phival);

                // add triangle
                phi_connect.push_back(tri_con_tmp);

                // debug
                /*
                if(rankvalues==0){
                    std::cout << "v1-x:"  << e12.x() << " y:" << e12.y() << " z:" << e12.z() << std::endl;
                    std::cout << "v2-x:"  << e13.x() << " y:" << e13.y() << " z:" << e13.z() << std::endl;
                    std::cout << "nor-x:" << nor.x() << " y:" << nor.y() << " z:" << nor.z() << std::endl;
                    std::cout << "vcheck-x:" << vcheck_tmp.x() << " y:" << vcheck_tmp.y() << " z:" << vcheck_tmp.z() << std::endl;
                    std::cout << "dot_val:"  << dot_val << std::endl;
                    std::cout << std::endl;
                }
                */
            }

            
            // 4 points case (Need to add two triangle)
            else if(phi_tmp_vertex.size() == 4){
                assert(phi_pos.size() == 2 and phi_neg.size() == 2);
                std::array<int, 4> vertex_4_index{};

                for(int itmp=0;itmp < phi_tmp_vertex.size() ; itmp++){
                    if (vertex_to_local_index.find(phi_tmp_vertex[itmp]) != vertex_to_local_index.end()) {
                        vertex_4_index[itmp] = vertex_to_local_index[phi_tmp_vertex[itmp]];
                    } else {
                        phi_vertex.push_back(phi_tmp_vertex[itmp]);
                        vertex_4_index[itmp] = phi_vertex.size() - 1;
                        vertex_to_local_index[phi_tmp_vertex[itmp]] = ver_loc_index_cur;
                        ver_loc_index_cur++;
                    }
                }

                int tmp_mark = compute_diagonal_direction(phi_tmp_vertex);

                // Add two triangle here
                for(int itmp=1;itmp<4;itmp++){
                    if(itmp == tmp_mark){
                        continue;
                    }
                    // Make sure they are empty
                    std::vector<int> tri_con_tmp;

                    // vertex 0
                    tri_con_tmp.push_back( vertex_4_index[0] );
                    // vertex tmp_mark
                    tri_con_tmp.push_back( vertex_4_index[tmp_mark] );
                    // vertex itmp
                    tri_con_tmp.push_back( vertex_4_index[itmp] );

                    // if the triangle is degenerated, skip it
                    if(tri_con_tmp[0] == tri_con_tmp[1] or
                       tri_con_tmp[0] == tri_con_tmp[2] or
                       tri_con_tmp[1] == tri_con_tmp[2]){
                        continue;
                    }

                    // Fix winding problem
                    compute_triangle_winding_in_tetrahedron(phi_vertex, phi_tet_vertex, tri_con_tmp, phival);

                    // add triangle
                    phi_connect.push_back(tri_con_tmp);
                }
            }
            
        }
    }

    void write_interface_txt(	std::vector<double>& vx,	\
                    std::vector<double>& vy,	\
                    std::vector<double>& vz,	\
                    std::vector<int>& color )
    {
        std::string interface_name = "phi0/tri"+std::to_string(step_num)+".dat";
        std::ofstream stFile(interface_name);

        for(int i=0;i<vx.size();i++){
            stFile << std::fixed << std::setprecision(10) << vx[i] << " " << vy[i] << " " << vz[i] << " " << color[i] << "\n";
        }
        stFile.close();
    }

    // clean triangle
    void clean_triangle(const std::vector<int> &c1_tmp,
                        const std::vector<int> &c2_tmp,
                        const std::vector<int> &c3_tmp,
                        std::vector<int> &c1_res,
                        std::vector<int> &c2_res,
                        std::vector<int> &c3_res,
                        const std::vector<double>&vx_un_res,
                        const std::vector<double>&vy_un_res,
                        const std::vector<double>&vz_un_res,
                        std::vector<double>&vx_res,
                        std::vector<double>&vy_res,
                        std::vector<double>&vz_res)
    {
        std::vector<int>     c1_un_res;
        std::vector<int>     c2_un_res;
        std::vector<int>     c3_un_res;

        std::vector<int> color;
        std::vector<int> color_num;
        std::vector<int> ver_map;	
        ver_map.resize(vx_un_res.size());

        int con_glo_index_cur = 0;
        std::string hash_tmp;	
        std::unordered_map<std::string, int> con_glo_index_map;
        details::Point3DMap 

        int rankvalues=details::MPI_rank(MPI_COMM_WORLD);

        // Triangle cleaning
        if(rankvalues == 0){
            for(int itmp=0;itmp<c1_tmp.size();itmp++){
                int c1_index = c1_tmp[itmp];
                int c2_index = c2_tmp[itmp];
                int c3_index = c3_tmp[itmp];

                if( c1_index == c2_index || c1_index == c3_index || c2_index == c3_index ){
                    continue;
                }
                else{
                    hash_con(c1_index,c2_index,c3_index,hash_tmp);
                    auto hash_sear = con_glo_index_map.find(hash_tmp);
                    if( hash_sear == con_glo_index_map.end() ){
                        c1_un_res.push_back( c1_index );
                        c2_un_res.push_back( c2_index );
                        c3_un_res.push_back( c3_index );
                        con_glo_index_map[hash_tmp] = con_glo_index_cur;
                        con_glo_index_cur++;
                    }
                }	
            }

            std::vector<std::set<int>>   node_connect(vx_un_res.size(), std::set<int>());
            for(int iele = 0;iele<c1_un_res.size();iele++){
                int c1_index = c1_un_res[iele];
                int c2_index = c2_un_res[iele];
                int c3_index = c3_un_res[iele];
                
                // edge c1-c2
                node_connect[c1_index].insert(c2_index);
                node_connect[c2_index].insert(c1_index);
                
                // edge c2-c3
                node_connect[c2_index].insert(c3_index);
                node_connect[c3_index].insert(c2_index);
                
                // edge c3-c1
                node_connect[c3_index].insert(c1_index);
                node_connect[c1_index].insert(c3_index);
            }

            cal_color(node_connect,color,color_num);
            for(int i=0;i<color_num.size();i++){
                std::cout<<"    Surface color "<<i<<" number: "<<color_num[i]<<std::endl;
            }
            write_interface_txt(vx_un_res,vy_un_res,vz_un_res,color);

            int num_tol     = 50;
            int clean_index = 0;
            for(int i=0;i<node_connect.size();i++){
                int color_nk = color_num[color[i]-1];
                //std::cout<<"Index "<<i<<" number: "<<color_nk<<std::endl;
                if(color_nk >= num_tol){
                    vx_res.push_back(vx_un_res[i]);
                    vy_res.push_back(vy_un_res[i]);
                    vz_res.push_back(vz_un_res[i]);
                    ver_map[i] = clean_index;
                    clean_index++;
                }
                else{
                    ver_map[i] = -1;
                }
            }

            for(int iele = 0;iele<c1_un_res.size();iele++){
                int c1_index = c1_un_res[iele];
                int c2_index = c2_un_res[iele];
                int c3_index = c3_un_res[iele];

                int v1_vermap_val = ver_map[c1_index];
                int v2_vermap_val = ver_map[c2_index];
                int v3_vermap_val = ver_map[c3_index];
            
                if( v1_vermap_val == -1 or v2_vermap_val == -1 or v3_vermap_val == -1 ){
                    continue;
                }
                else{
                    c1_res.push_back(v1_vermap_val);
                    c2_res.push_back(v2_vermap_val);
                    c3_res.push_back(v3_vermap_val);
                }
            }
        }

        
        details::MPI_broadcast(MPI_COMM_WORLD,vx_res);
        details::MPI_broadcast(MPI_COMM_WORLD,vy_res);
        details::MPI_broadcast(MPI_COMM_WORLD,vz_res);

        details::MPI_broadcast(MPI_COMM_WORLD,c1_res);
        details::MPI_broadcast(MPI_COMM_WORLD,c2_res);
        details::MPI_broadcast(MPI_COMM_WORLD,c3_res);

    }

    // combine triangles
    void combine_triangle(const std::vector<Point3D> &phi_vertex,
                          const std::vector<std::vector<int>> &phi_connect,
                          std::vector<double> &vx_res,
                          std::vector<double> &vy_res,
                          std::vector<double> &vz_res,
                          std::vector<int> &c1_res,
                          std::vector<int> &c2_res,
                          std::vector<int> &c3_res) {
        std::string hash_tmp;
        int ver_glo_index_cur = 0;

        // std::unordered_map<std::string, int> ver_glo_index_map;	
        details::Point3DMap<size_t> ver_glo_index_map;

        std::vector<double>  vx_un_res;
        std::vector<double>  vy_un_res;
        std::vector<double>  vz_un_res;

        std::vector<int>     c1_tmp;
        std::vector<int>     c2_tmp;
        std::vector<int>     c3_tmp;

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
        for(int iphi=0;iphi<phi_vertex.size();iphi++)
        {
            phi_cx_loc.push_back( phi_vertex[iphi].x() );
            phi_cy_loc.push_back( phi_vertex[iphi].y() );
            phi_cz_loc.push_back( phi_vertex[iphi].z() );
        }

        // MPI communication 
        // MPI::gather(MPI_COMM_WORLD,phi_cx_loc,phi_cx_glo);
        // MPI::gather(MPI_COMM_WORLD,phi_cy_loc,phi_cy_glo);
        // MPI::gather(MPI_COMM_WORLD,phi_cz_loc,phi_cz_glo);

        // MPI::broadcast(MPI_COMM_WORLD,phi_cx_glo);
        // MPI::broadcast(MPI_COMM_WORLD,phi_cy_glo);
        // MPI::broadcast(MPI_COMM_WORLD,phi_cz_glo);
        details::MPI_allgatherv(MPI_COMM_WORLD,phi_cx_loc,phi_cx_glo);
        details::MPI_allgatherv(MPI_COMM_WORLD,phi_cy_loc,phi_cy_glo);
        details::MPI_allgatherv(MPI_COMM_WORLD,phi_cz_loc,phi_cz_glo);

        for(int itmp = 0; itmp < phi_cx_glo.size(); itmp++) {
            Point3D point {phi_cx_glo[itmp],phi_cy_glo[itmp],phi_cz_glo[itmp]};
            auto it = ver_glo_index_map.find(point);
            if (it == ver_glo_index_map.end()) {
                ver_glo_index_map[point] = ver_glo_index_cur;
                ver_glo_index_cur++;

                // Add new vertex to global vector
                vx_un_res.push_back( phi_cx_glo[itmp] );
                vy_un_res.push_back( phi_cy_glo[itmp] );
                vz_un_res.push_back( phi_cz_glo[itmp] );
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

        //Following code do MPI communication and change the old index in connect to new index
        std::vector<int> tri_a_index_loc;
        std::vector<int> tri_b_index_loc;
        std::vector<int> tri_c_index_loc;
        std::vector<int> tri_a_index_tmp;
        std::vector<int> tri_b_index_tmp;
        std::vector<int> tri_c_index_tmp;

        // copy to local vector
        for(int iphi = 0; iphi < phi_connect.size(); iphi++) {
            tri_a_index_loc.push_back(phi_connect[iphi][0]);
            tri_b_index_loc.push_back(phi_connect[iphi][1]);
            tri_c_index_loc.push_back(phi_connect[iphi][2]);
        }

        // transform old index to new index on each local processor first
        for(int itmp = 0; itmp < tri_a_index_loc.size(); itmp++){
            auto it = ver_glo_index_map.find(phi_vertex[tri_a_index_loc[itmp]]);
            if (it != ver_glo_index_map.end()) {
                tri_a_index_tmp.push_back( it->second );
            }
            else{
                printf("Error: can not find the vertex in local processor");
            }
            // double tmp_cx = phi_vertex[tri_a_index_loc[itmp]].x();
            // double tmp_cy = phi_vertex[tri_a_index_loc[itmp]].y();
            // double tmp_cz = phi_vertex[tri_a_index_loc[itmp]].z();

            // hash_cor(tmp_cx,tmp_cy,tmp_cz,hash_tmp);
            // auto hash_sear = ver_glo_index_map.find(hash_tmp);
            // if( hash_sear != ver_glo_index_map.end() ){
            //     tri_a_index_tmp.push_back( hash_sear->second );
            //     //printf("before:%.15e, %.15e, %.15e\n",tmp_cx,tmp_cy,tmp_cz);
            //     //printf("after:%.15e, %.15e, %.15e\n",vx_un_res[hash_sear->second],vy_un_res[hash_sear->second],vz_un_res[hash_sear->second]);
            //     //std::cout<<std::endl;
            // }
            // else{
            //     //std::cout << "rank: " << rankvalues << " " << "index: " << tri_a_index_loc[itmp] << " " <<  tmp_cx << " " << tmp_cy << " " << tmp_cz << std::endl;
            //     std::cout << "Search in ver_glo_index_map not found. Must be something wrong." << std::endl;
            // }
        }

        for(int itmp=0;itmp<tri_b_index_loc.size();itmp++){
            auto it = ver_glo_index_map.find(phi_vertex[tri_b_index_loc[itmp]]);
            if (it != ver_glo_index_map.end()) {
                tri_b_index_tmp.push_back( it->second );
            }
            else{
                printf("Error: can not find the vertex in local processor");
            }
            // double tmp_cx = phi_vertex[tri_b_index_loc[itmp]].x();
            // double tmp_cy = phi_vertex[tri_b_index_loc[itmp]].y();
            // double tmp_cz = phi_vertex[tri_b_index_loc[itmp]].z();

            // hash_cor(tmp_cx,tmp_cy,tmp_cz,hash_tmp);
            // auto hash_sear = ver_glo_index_map.find(hash_tmp);
            // if( hash_sear != ver_glo_index_map.end() ){
            //     tri_b_index_tmp.push_back( hash_sear->second );
            //     //printf("before:%.15e, %.15e, %.15e\n",tmp_cx,tmp_cy,tmp_cz);
            //     //printf("after:%.15e, %.15e, %.15e\n",vx_un_res[hash_sear->second],vy_un_res[hash_sear->second],vz_un_res[hash_sear->second]);
            //     //std::cout<<std::endl;
            // }
            // else{
            //     //std::cout << "rank: " << rankvalues << " " << "index: " << tri_b_index_loc[itmp] << " " <<  tmp_cx << " " << tmp_cy << " " << tmp_cz << std::endl;
            //     std::cout << "Search in ver_glo_index_map not found. Must be something wrong." << std::endl;
            // }
        }

        for(int itmp=0;itmp<tri_c_index_loc.size();itmp++){
            auto it = ver_glo_index_map.find(phi_vertex[tri_c_index_loc[itmp]]);
            if (it != ver_glo_index_map.end()) {
                tri_c_index_tmp.push_back( it->second );
            }
            else{
                printf("Error: can not find the vertex in local processor");
            }
            // double tmp_cx = phi_vertex[tri_c_index_loc[itmp]].x();
            // double tmp_cy = phi_vertex[tri_c_index_loc[itmp]].y();
            // double tmp_cz = phi_vertex[tri_c_index_loc[itmp]].z();

            // hash_cor(tmp_cx,tmp_cy,tmp_cz,hash_tmp);
            // auto hash_sear = ver_glo_index_map.find(hash_tmp);
            // if( hash_sear != ver_glo_index_map.end() ){
            //     tri_c_index_tmp.push_back( hash_sear->second );
            //     //printf("before:%.15e, %.15e, %.15e\n",tmp_cx,tmp_cy,tmp_cz);
            //     //printf("after:%.15e, %.15e, %.15e\n",vx_un_res[hash_sear->second],vy_un_res[hash_sear->second],vz_un_res[hash_sear->second]);
            //     //std::cout<<std::endl;
            // }
            // else{
            //     //std::cout << "rank: " << rankvalues << " " << "index: " << tri_c_index_loc[itmp] << " " <<  tmp_cx << " " << tmp_cy << " " << tmp_cz << std::endl;
            //     std::cout << "Search in ver_glo_index_map not found. Must be something wrong." << std::endl;
            // }
        }

        
        // MPI communication 
        details::MPI_gather(MPI_COMM_WORLD,tri_a_index_tmp,c1_tmp);
        details::MPI_gather(MPI_COMM_WORLD,tri_b_index_tmp,c2_tmp);
        details::MPI_gather(MPI_COMM_WORLD,tri_c_index_tmp,c3_tmp);

        clean_triangle(c1_tmp, c2_tmp, c3_tmp,
                       c1_res, c2_res, c3_res,
                       vx_un_res, vy_un_res, vz_un_res,
                       vx_res, vy_res, vz_res);

    }

    // compute nodal normal vector
    void compute_normal(const std::vector<double> &vx_res,
                        const std::vector<double> &vy_res,
                        const std::vector<double> &vz_res,
                        const std::vector<int> &c1_res,
                        const std::vector<int> &c2_res,
                        const std::vector<int> &c3_res,
                        std::vector<double> &vertex_nor_x,
                        std::vector<double> &vertex_nor_y,
                        std::vector<double> &vertex_nor_z,
                        std::vector<double> &vertex_nor_num,
                        std::vector<double> &vertex_nor_area) {
        // Resize vertex_nor size
        vertex_nor_x.assign(vx_res.size(), 0.0);
        vertex_nor_y.assign(vx_res.size(), 0.0);
        vertex_nor_z.assign(vx_res.size(), 0.0);
        vertex_nor_num.assign(vx_res.size(), 0.0);
        vertex_nor_area.assign(vx_res.size(), 0.0);

        int rankvalues = details::MPI_rank(MPI_COMM_WORLD);

        // Normal vector
        if(rankvalues==0) {
            for(int ielem = 0; ielem < c1_res.size(); ielem++){
                int c1_index = c1_res[ielem];
                int c2_index = c2_res[ielem];
                int c3_index = c3_res[ielem];
                
                // Get triangle vertex coordinate
                Point3D v1_tmp( vx_res[c1_index],vy_res[c1_index],vz_res[c1_index] );
                Point3D v2_tmp( vx_res[c2_index],vy_res[c2_index],vz_res[c2_index] );
                Point3D v3_tmp( vx_res[c3_index],vy_res[c3_index],vz_res[c3_index] );

                // Calculate the normal vector for triangle
                double unit_scale = 1e6;
                Point3D e12 = (v2_tmp - v1_tmp)*unit_scale;
                Point3D e13 = (v3_tmp - v1_tmp)*unit_scale;
                Point3D nor = e12.cross(e13);

                double nor_mag = nor.norm();
                double tri_area= nor_mag/2.0;
                nor = nor/(nor_mag + 1e-30);

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
                vertex_nor_num[c1_index]  += 1.0;
                vertex_nor_num[c2_index]  += 1.0;
                vertex_nor_num[c3_index]  += 1.0;

                vertex_nor_area[c1_index] += tri_area;
                vertex_nor_area[c2_index] += tri_area;
                vertex_nor_area[c3_index] += tri_area;
                
            }

            // Average the normal vector
            for(int itmp=0;itmp<vertex_nor_x.size();itmp++){
                if(vertex_nor_num[itmp] > 0){
                    vertex_nor_x[itmp] = vertex_nor_x[itmp]/(vertex_nor_num[itmp]*1.0);
                    vertex_nor_y[itmp] = vertex_nor_y[itmp]/(vertex_nor_num[itmp]*1.0);
                    vertex_nor_z[itmp] = vertex_nor_z[itmp]/(vertex_nor_num[itmp]*1.0);

                    // normalize normal
                    double mag_tol = 1e-30;
                    double nor_mag = std::sqrt( 	vertex_nor_x[itmp]*vertex_nor_x[itmp] + \
                                    vertex_nor_y[itmp]*vertex_nor_y[itmp] + \
                                    vertex_nor_z[itmp]*vertex_nor_z[itmp] ) + mag_tol;
                    
                    vertex_nor_x[itmp] = vertex_nor_x[itmp]/nor_mag;
                    vertex_nor_y[itmp] = vertex_nor_y[itmp]/nor_mag;
                    vertex_nor_z[itmp] = vertex_nor_z[itmp]/nor_mag;
                }
                else{
                    std::cout<<"Zero normal vectors find !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
                }
            }
        }

        details::MPI_broadcast(MPI_COMM_WORLD,vertex_nor_x);
        details::MPI_broadcast(MPI_COMM_WORLD,vertex_nor_y);
        details::MPI_broadcast(MPI_COMM_WORLD,vertex_nor_z);

        details::MPI_broadcast(MPI_COMM_WORLD,vertex_nor_num);
        details::MPI_broadcast(MPI_COMM_WORLD,vertex_nor_area);

    }

    // Laplacian smoothing
    void laplace_smooth(	std::vector<double>&  vx_res,	\
                std::vector<double>&  vy_res,	\
                std::vector<double>&  vz_res,	\
                std::vector<int>&     c1_res,	\
                std::vector<int>&     c2_res,	\
                std::vector<int>&     c3_res)
    {
        {
            int lap_num = 0;
            
            std::vector<std::set<int>>   node_con(vx_res.size(), std::set<int>());
            for(int iele = 0;iele<c1_res.size();iele++){
                int c1_index = c1_res[iele];
                int c2_index = c2_res[iele];
                int c3_index = c3_res[iele];
                
                // edge c1-c2
                node_con[c1_index].insert(c2_index);
                node_con[c2_index].insert(c1_index);
                
                // edge c2-c3
                node_con[c2_index].insert(c3_index);
                node_con[c3_index].insert(c2_index);
                
                // edge c3-c1
                node_con[c3_index].insert(c1_index);
                node_con[c1_index].insert(c3_index);
            }

            std::set<int>::iterator node_it;
            std::vector<double> vx_lap(vx_res.size(),  0.0);
            std::vector<double> vy_lap(vx_res.size(),  0.0);
            std::vector<double> vz_lap(vx_res.size(),  0.0);
            std::vector<double> ver_num(vx_res.size(), 0.0);
            
            for(int ilap = 0; ilap<lap_num;ilap++){
                
                std::fill(vx_lap.begin(),  vx_lap.end(),  0.0);
                std::fill(vy_lap.begin(),  vy_lap.end(),  0.0);
                std::fill(vz_lap.begin(),  vz_lap.end(),  0.0);
                std::fill(ver_num.begin(), ver_num.end(), 0.0);
        
                for(int inode=0;inode<vx_res.size();inode++){
                    for (node_it = node_con[inode].begin(); node_it != node_con[inode].end(); ++node_it) {
                        vx_lap[inode] += vx_res[*node_it];
                        vy_lap[inode] += vy_res[*node_it];
                        vz_lap[inode] += vz_res[*node_it];
                        ver_num[inode]+= 1.0;;
                    }
                }
                
                for(int inode=0;inode<vx_res.size();inode++){
                    if(ver_num[inode] < 0.5){
                        std::cout<<"No vertex connect to current vertex. Something wrong. "<<std::endl;
                    }
                    else{
                        vx_res[inode] = vx_lap[inode]/ver_num[inode];
                        vy_res[inode] = vy_lap[inode]/ver_num[inode];
                        vz_res[inode] = vz_lap[inode]/ver_num[inode];
                    }
                }
            }

        }
        details::MPI_broadcast(MPI_COMM_WORLD,vx_res);
        details::MPI_broadcast(MPI_COMM_WORLD,vy_res);
        details::MPI_broadcast(MPI_COMM_WORLD,vz_res);
        
    }

    // Write out phi=0 surface triangulation
    void write_triangulation(std::vector<double>&  vx_res,	\
                std::vector<double>&  vy_res,	\
                std::vector<double>&  vz_res,	\
                std::vector<double>&  nx_res,  \
                            std::vector<double>&  ny_res,  \
                            std::vector<double>&  nz_res,  \
                std::vector<int>&     c1_res,	\
                std::vector<int>&     c2_res,	\
                std::vector<int>&     c3_res)
    {
        int rankvalues=dolfin::MPI::rank(MPI_COMM_WORLD);

        if(rankvalues==0){
            std::string name = "sur/vertex"+std::to_string(step_num)+".dat";
            std::ofstream pFile(name);
            for(int itmp=0;itmp<vx_res.size();itmp++){
                pFile << std::fixed << std::setprecision(10) << vx_res[itmp] << " " << vy_res[itmp] << " " << vz_res[itmp] << "\n";
            }
            pFile.close();
        }

        if(rankvalues==0){
                    std::string name = "sur/normal"+std::to_string(step_num)+".dat";
                    std::ofstream pFile(name);
                    for(int itmp=0;itmp<nx_res.size();itmp++){
                            pFile << std::fixed << std::setprecision(10) << nx_res[itmp] << " " << ny_res[itmp] << " " << nz_res[itmp] << "\n";
                    }
                    pFile.close();
            }

        if(rankvalues==0){
            std::string name = "sur/connect"+std::to_string(step_num)+".dat";
                    std::ofstream pFile(name);
                    for(int itmp=0;itmp<c1_res.size();itmp++){
                            pFile << c1_res[itmp] << " " << c2_res[itmp] << " " << c3_res[itmp] << "\n";
                    }
                    pFile.close();

        }
    }

    void filter_point(std::vector<double> &vx_res,
                      std::vector<double> &vy_res,
                      std::vector<double> &vz_res,
                      std::vector<double> &vertex_nor_x,
                      std::vector<double> &vertex_nor_y,
                      std::vector<double> &vertex_nor_z,
                      const std::vector<double> &vertex_nor_num,
                      const std::vector<double> &vertex_nor_area ) {
        int num = vx_res.size();

        std::vector<double> vx_new;
        std::vector<double> vy_new;
        std::vector<double> vz_new;

        std::vector<double> nx_new;
            std::vector<double> ny_new;
            std::vector<double> nz_new;

        for(int i = 0; i < num; i++){
            if( vertex_nor_num[i] > 2.5 ){
                vx_new.push_back(vx_res[i]);
                vy_new.push_back(vy_res[i]);
                vz_new.push_back(vz_res[i]);

                nx_new.push_back(vertex_nor_x[i]);
                            ny_new.push_back(vertex_nor_y[i]);
                            nz_new.push_back(vertex_nor_z[i]);
            }
        }

        vx_res.swap( vx_new );
        vy_res.swap( vy_new );
        vz_res.swap( vz_new );

        vertex_nor_x.swap( nx_new );
        vertex_nor_y.swap( ny_new );
        vertex_nor_z.swap( nz_new );

    }

    // Compute levelset sign distance function here
    void compute_levelset_distance(const TetrahedronMesh &mesh,
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

        // Start to calculate the distance between point and point cloud (bounding box)
        // Copy to point vector
        for(int iphi = 0; iphi < vx_res.size(); iphi++) {
            phi_inter_glo.push_back( Point(vx_res[iphi],vy_res[iphi],vz_res[iphi]) );
        }

        bool ismaster = details::MPI_rank(MPI_COMM_WORLD) == 0;
        if(ismaster and phi_inter_glo.empty()) {
            std::cout << "The triangulization of phi=0 has zero point. "
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
        for(int iphi = 0; iphi < phid0_vec.size(); iphi++) {
            // Get the vertex index from dof index
            std::size_t phi_ind = d2v_map[iphi];

            // Get the coordinate of corresponding point
            // Point3D tmp_point(local_cor[3*phi_ind],local_cor[3*phi_ind+1],local_cor[3*phi_ind+2]);
            const auto& tmp_point = mesh.vertex[phi_ind];

            // Use bounding box to get the closest distance from point to point cloud
            std::pair<int,double> dist_pair;
            int    oct_tree_index;
            double oct_tree_dist;
            phi0_Bound_Box_Tree.search_point(tmp_point,oct_tree_index,oct_tree_dist);

            dist_pair  = std::make_pair(oct_tree_index,oct_tree_dist);		


            // Get the nearest point in the cloud
            Point3D near_point_cloud = phi_inter_glo[dist_pair.first];

            // Calculate the averaged normal on vertex
            Point3D nor_vertex_cal{
                    vertex_nor_x[dist_pair.first],
                    vertex_nor_y[dist_pair.first],
                    vertex_nor_z[dist_pair.first]};

            // Get the line connecting vertex point and near point in cloud.

            // Calculate projection length over vertex normal direction
            double dist_proj = std::abs(nor_vertex_cal.dot(tmp_point - near_point_cloud));

            /*
            if( near_edge.dot(nor_vertex_cal) > 0.0 ){
                phi_sign =  1.0;
            }
            else{
                phi_sign = -1.0;
            }		
            */
            
            if(phid0_vec[iphi] > 0.0){
                phi_sign =  1.0;
            }
            else{
                phi_sign = -1.0;
            }

            // Change dropped point sign
            Vertex phi_ver = Vertex(*mesh, phi_ind);
            double phi_factor_val = phi_factor[ phi_ver.global_index() ] * 1.0;
            phi_sign = phi_sign*phi_factor_val;

            double phi_mag = std::abs( phid0_vec[iphi] );
            double phi_tol = 1.5*epslen_;
            
            if( phi_mag > phi_tol ){
                // Use distance to point cloud
                phid0_vec[iphi] = phi_sign*dist_pair.second;
            }
            else{
                // Use projected distance
                phid0_vec[iphi] = phi_sign*dist_proj;
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

        // Insert the solution to old function
        phid0->vector()->set_local(phid0_vec);
        phid0->vector()->apply("insert");
    }

    // tecplot communication computation
    void tec_map_compute(   std::vector<la_index>& local_ind_full,
                            std::vector<la_index>& local_ind_reduce,
                            std::vector<int>&      global_ind_reduce_loc_glo,
                            std::shared_ptr<Mesh>  mesh )
    {
            std::vector<int> global_ind_reduce_loc;

        int rankvalues = dolfin::MPI::rank(MPI_COMM_WORLD);        
            int ii=0;
            for (VertexIterator v(*mesh); !v.end(); ++v)
            {
                    ii++;
                    local_ind_full.push_back(ii-1);
                    int global_ind = v->global_index();
                    if(v->is_shared()){
                            auto sharing_id = v->sharing_processes();
                            sharing_id.insert(rankvalues);

                            if(rankvalues != *(sharing_id.begin()) ){
                                    continue;
                            }
                    }

                    local_ind_reduce.push_back(ii-1);
                    global_ind_reduce_loc.push_back(global_ind);
            }
            dolfin::MPI::gather<int>(MPI_COMM_WORLD,global_ind_reduce_loc,global_ind_reduce_loc_glo,0);
    };

    // combine distributed scalar data to master processor
    void tec_scalar_comm(   std::shared_ptr<Function>& p0,
                std::vector<double>&       p0_out,
                            std::vector<la_index>& local_ind_full,
                            std::vector<la_index>& local_ind_reduce,
                            std::vector<int>&      global_ind_reduce_loc_glo,
                std::vector<dolfin::la_index>& vertex2dof_map)
    {
        int rankvalues = dolfin::MPI::rank(MPI_COMM_WORLD);

            std::vector<double> p0_vec;
            std::size_t p0_shape = local_ind_full.size();
            p0_vec.resize(p0_shape);
            p0->vector()->get_local(p0_vec.data(), p0_shape, local_ind_full.data());

            std::vector<double> p0_loc;
            std::vector<double> p0_glo;
            p0_loc.resize(local_ind_reduce.size());

            for(int i=0;i<local_ind_reduce.size();i++){
                    auto dof_ind = vertex2dof_map[local_ind_reduce[i]];
                    p0_loc[i] = p0_vec[dof_ind];
            }
            dolfin::MPI::gather<double>(MPI_COMM_WORLD,p0_loc,p0_glo,0);

            if(rankvalues==0){
                    p0_out.resize(global_ind_reduce_loc_glo.size());

                    for(int i=0;i<global_ind_reduce_loc_glo.size();i++){
                            p0_out[global_ind_reduce_loc_glo[i]] = p0_glo[i];
                    }

                    //for(int i=0;i<p0_out.size();i++){
                    //        std::cout<<p0_out[i]<<std::endl;
                    //}
            }
            
    };

    // combine data to master processor
    void tec_phi_comm(	std::shared_ptr<Function>& phi0,
                std::vector<double>& phi0_vec,
                            std::vector<dolfin::la_index>& vertex2dof_map,
                            std::shared_ptr<Mesh>&  mesh)
    {
        /*
        std::vector<la_index> local_ind_full;
            std::vector<la_index> local_ind_reduce;
            std::vector<int>      global_ind_reduce_loc_glo;
        int rankvalues = dolfin::MPI::rank(MPI_COMM_WORLD);
        tec_map_compute(local_ind_full,local_ind_reduce,global_ind_reduce_loc_glo,mesh);
        tec_scalar_comm(phi0, phi0_vec,local_ind_full,local_ind_reduce,global_ind_reduce_loc_glo,vertex2dof_map);
        */

        // tec_scalar_comm(phi0, phi0_vec,this->tec_comm_local_ind_full,this->tec_comm_local_ind_reduce,this->tec_comm_global_ind_reduce_loc_glo,vertex2dof_map);
    };

    // Compute edge connectivity
    void compute_connect( std::vector<std::set<int>>& node_connect_glo, std::shared_ptr<Mesh>& mesh)
    {

        // Get some some mesh
        const std::size_t tdim = mesh->topology().dim();
        const std::size_t gdim = mesh->geometry().dim();
        const std::size_t num_local_vertices  = mesh->num_entities(0);
        const std::size_t num_global_vertices = mesh->num_entities_global(0);

        std::vector<int> node_connect_loc_a;
        std::vector<int> node_connect_loc_b;
        std::vector<int> node_connect_glo_a;
        std::vector<int> node_connect_glo_b;
        std::vector<std::set<int>> node_connect_loc(num_global_vertices, std::set<int>());

        node_connect_glo.resize(num_global_vertices);
        int rankvalues = dolfin::MPI::rank(MPI_COMM_WORLD);

        // Compute connectivity
        int cell_ind = 0;
        std::vector<int> cell_vec(4,0);
        for (CellIterator cell(*mesh); !cell.end(); ++cell)
        {
            int vertex_ind=0;
            for (VertexIterator v(*cell); !v.end(); ++v)
            {
                int global_ind = v->global_index();
                cell_vec[vertex_ind]=global_ind;
                vertex_ind++;
            }

            for (int i=0;i<4;i++){
                for(int j=0;j<4;j++){
                    if(i==j) continue;

                    node_connect_loc[cell_vec[i]].insert( cell_vec[j] );
                    node_connect_loc[cell_vec[j]].insert( cell_vec[i] );
                }
            }
            cell_ind++;
        }

        std::set<int>::iterator node_it;
        for(int i=0;i<num_global_vertices;i++){
            for (node_it = node_connect_loc[i].begin(); node_it != node_connect_loc[i].end(); ++node_it) {
                int val = *node_it;
                node_connect_loc_a.push_back(i);
                node_connect_loc_b.push_back(val);
            }
        }

        MPI::gather(MPI_COMM_WORLD,node_connect_loc_a,node_connect_glo_a);
        MPI::gather(MPI_COMM_WORLD,node_connect_loc_b,node_connect_glo_b);

        if(rankvalues==0){
            for(int i=0;i<node_connect_glo_a.size();i++){
                node_connect_glo[ node_connect_glo_a[i] ].insert( node_connect_glo_b[i] );
            }
        }

    }

    void redistance_prep( std::shared_ptr<Mesh>& mesh ) {
        //tec_map_compute(this->tec_comm_local_ind_full,this->tec_comm_local_ind_reduce,this->tec_comm_global_ind_reduce_loc_glo,mesh);
        compute_connect(this->elem_connect,mesh);
    }

    void compute_vol_color(	std::shared_ptr<Mesh>& mesh, \
                std::shared_ptr<Function> phid0, \
                std::vector<int>& phi_sign, \
                std::vector<std::size_t>& d2v_map, \
                std::vector<dolfin::la_index>& v2d_map) {
        std::vector<int> color;
        std::vector<int> color_num;

        std::vector<double> phi0_vec;

        int rankvalues = details::MPI_rank(MPI_COMM_WORLD);

        // Combine all phi value to master processor (MPI_rank == 0)
        // tec_phi_comm(phid0, phi0_vec, v2d_map, mesh);

        if(rankvalues==0) {
            
            cal_color(this->elem_connect, phi0_vec, color, color_num);
            for(int i = 0; i < color_num.size(); i++) {
                std::cout << "    Volume color " << i << " number: " << color_num[i] <<std::endl;
            }
            
            phi_sign.resize(phi0_vec.size());

            int num_tol = 100;
            for(int i=0;i<phi_sign.size();i++) {
                int color_nk = color_num[color[i]-1];
                if(color_nk >= num_tol){
                    phi_sign[i] = 1;
                }
                else{
                    phi_sign[i] = -1;
                }
            }
        }

        details::MPI_broadcast(MPI_COMM_WORLD,phi_sign);
    }

    void phi_sign_init(	std::vector<int> &phi_sign, \
                std::shared_ptr<Mesh>& mesh)
    {
        const std::size_t num_global_vertices = mesh->num_entities_global(0);

        phi_sign.resize(num_global_vertices);
        std::fill(phi_sign.begin(),phi_sign.end(),1);
    }

    void geo_redistance(	std::shared_ptr<Mesh>& mesh, \
                std::shared_ptr<Function> phid0, \
                std::vector<std::size_t>& d2v_map, \
                std::vector<dolfin::la_index>& v2d_map )
    {
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

        std::vector<double>  vx_res;
        std::vector<double>  vy_res;
        std::vector<double>  vz_res;

        std::vector<int>     c1_res;
        std::vector<int>     c2_res;
        std::vector<int>     c3_res;

        std::vector<double>  vertex_nor_x;
        std::vector<double>  vertex_nor_y;
        std::vector<double>  vertex_nor_z;

        std::vector<double>  vertex_nor_num;
            std::vector<double>  vertex_nor_area;

        std::vector<int>     phi_sign;

        bool ismaster = details::MPI_rank(MPI_COMM_WORLD) == 0;


        // Triangulation of the phi=0
        add_triangle(mesh, phid0, phi_vertex, phi_connect);


        //if(ismaster) info("Start to combine triangle.");
        // Combine triangle from different processor and remove duplicate
        combine_triangle(phi_vertex, phi_connect,\
                vx_res, vy_res, vz_res, \
                c1_res, c2_res, c3_res);


        //if(ismaster) info("Start to compute normal.");
        // Compute normal vector at vertex
        compute_normal(	vx_res,	vy_res, vz_res,				\
                c1_res,	c2_res,	c3_res,				\
                vertex_nor_x, vertex_nor_y, vertex_nor_z,       \
                vertex_nor_num, vertex_nor_area);

        // Laplace smoothing
        //laplace_smooth(vx_res, vy_res, vz_res, c1_res, c2_res, c3_res);     

        // debug use
        // write_triangulation(vx_res, vy_res, vz_res,\
        //             vertex_nor_x, vertex_nor_y, vertex_nor_z,\
        //             c1_res, c2_res, c3_res );

        filter_point(vx_res, vy_res, vz_res,
                     vertex_nor_x, vertex_nor_y, vertex_nor_z,
                     vertex_nor_num, vertex_nor_area);

        // Compute volume color
        if( vol_color_flag == 1){
            compute_vol_color(mesh, phid0, phi_sign, d2v_map, v2d_map);
        }
        else{
            phi_sign_init(phi_sign, mesh);
        }


        //if(ismaster) info("Start to compute levelset distance.");
        // Levelset redistancing calculation
        compute_levelset_distance(mesh, phid0, phi_sign, d2v_map,\
                    vx_res, vy_res, vz_res, \
                    vertex_nor_x, vertex_nor_y, vertex_nor_z);

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
}
#endif // __REDIS_H__