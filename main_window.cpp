/********************************************************************
 * Copyright (C) 2015 Liangliang Nan <liangliang.nan@gmail.com>
 * https://3d.bk.tudelft.nl/liangliang/
 *
 * This file is part of Easy3D. If it is useful in your research/work,
 * I would be grateful if you show your appreciation by citing it:
 * ------------------------------------------------------------------
 *      Liangliang Nan.
 *      Easy3D: a lightweight, easy-to-use, and efficient C++ library
 *      for processing and rendering 3D data.
 *      Journal of Open Source Software, 6(64), 3255, 2021.
 * ------------------------------------------------------------------
 *
 * Easy3D is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License Version 3
 * as published by the Free Software Foundation.
 *
 * Easy3D is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 ********************************************************************/


#include "main_window.h"

#include <string>
#include <iostream>

#include <QMutex>
#include <QFileDialog>
#include <QDropEvent>
#include <QMimeData>
#include <QSettings>
#include <QMessageBox>
#include <QColorDialog>
#include <QCoreApplication>
#include <QLabel>
#include <QPushButton>
#include <QProgressBar>

#include <easy3d/core/version.h>
#include <easy3d/core/surface_mesh.h>
#include <easy3d/core/graph.h>
#include <easy3d/core/point_cloud.h>
#include <easy3d/core/poly_mesh.h>
#include <easy3d/core/random.h>
#include <easy3d/core/surface_mesh_builder.h>
#include <easy3d/renderer/setting.h>
#include <easy3d/renderer/camera.h>
#include <easy3d/renderer/renderer.h>
#include <easy3d/renderer/clipping_plane.h>
#include <easy3d/renderer/drawable_lines.h>
#include <easy3d/renderer/drawable_points.h>
#include <easy3d/renderer/key_frame_interpolator.h>
#include <easy3d/renderer/drawable_triangles.h>
#include <easy3d/renderer/manipulator.h>
#include <easy3d/renderer/transform.h>
#include <easy3d/fileio/point_cloud_io.h>
#include <easy3d/fileio/graph_io.h>
#include <easy3d/fileio/surface_mesh_io.h>
#include <easy3d/fileio/poly_mesh_io.h>
#include <easy3d/fileio/ply_reader_writer.h>
#include <easy3d/fileio/point_cloud_io_ptx.h>
#include <easy3d/fileio/translator.h>
#include <easy3d/algo/point_cloud_normals.h>
#include <easy3d/algo/surface_mesh_components.h>
#include <easy3d/algo/surface_mesh_topology.h>
#include <easy3d/algo/surface_mesh_triangulation.h>
#include <easy3d/algo/surface_mesh_tetrahedralization.h>
#include <easy3d/algo/surface_mesh_subdivision.h>
#include <easy3d/algo/surface_mesh_geodesic.h>
#include <easy3d/algo/surface_mesh_stitching.h>
#include <easy3d/algo/surface_mesh_enumerator.h>
#include <easy3d/algo/surface_mesh_polygonization.h>
#include <easy3d/algo/surface_mesh_geometry.h>
#include <easy3d/algo_ext/surfacer.h>
#include <easy3d/algo/delaunay_2d.h>
#include <easy3d/algo/delaunay_3d.h>
#include <easy3d/util/logging.h>
#include <easy3d/util/file_system.h>
#include <easy3d/util/stop_watch.h>
#include <easy3d/util/line_stream.h>
#include <easy3d/util/string.h>
#include <easy3d/fileio/resources.h>
#include <easy3d/renderer/vertex_array_object.h>

#include "paint_canvas.h"
#include "walk_through.h"
#include "yaml.h"

#include "dialogs/dialog_snapshot.h"
#include "dialogs/dialog_properties.h"
#include "dialogs/dialog_poisson_reconstruction.h"
#include "dialogs/dialog_surface_mesh_curvature.h"
#include "dialogs/dialog_surface_mesh_sampling.h"
#include "dialogs/dialog_point_cloud_normal_estimation.h"
#include "dialogs/dialog_point_cloud_ransac_primitive_extraction.h"
#include "dialogs/dialog_point_cloud_simplification.h"
#include "dialogs/dialog_gaussian_noise.h"
#include "dialogs/dialog_surface_mesh_fairing.h"
#include "dialogs/dialog_surface_mesh_from_text.h"
#include "dialogs/dialog_surface_mesh_hole_filling.h"
#include "dialogs/dialog_surface_mesh_parameterization.h"
#include "dialogs/dialog_surface_mesh_remeshing.h"
#include "dialogs/dialog_surface_mesh_smoothing.h"
#include "dialogs/dialog_surface_mesh_simplification.h"
#include "dialogs/dialog_walk_through.h"

#include "widgets/widget_global_setting.h"
#include "widgets/widget_drawable_points.h"
#include "widgets/widget_drawable_lines.h"
#include "widgets/widget_drawable_triangles.h"

#include <ui_main_window.h>

#include <easy3d/core/principal_axes.h>
#include <easy3d/algo_ext/surfacer.h>
#include <easy3d/kdtree/kdtree_search_eth.h>
#include <easy3d/algo/point_cloud_simplification.h>
#include <easy3d/algo/point_cloud_ransac.h>
#include <easy3d/renderer/texture.h>

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/utility.hpp"

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Alpha_shape_2.h>
#include <CGAL/Projection_traits_xy_3.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_hierarchy_2.h>
#include <CGAL/linear_least_squares_fitting_3.h>
#include <CGAL/convex_hull_2.h>

#if CGAL_VERSION_NR < 1041100000
#error CGAL 4.11 or above is required (due to the breaking change in CGAL 4.11). Please update your code.
#endif

typedef CGAL::Simple_cartesian<float>   Kernel;
typedef typename Kernel::Point_3        Point_3;
typedef typename Kernel::Vector_3       Vector_3;
typedef typename Kernel::Plane_3        Plane_3;
typedef typename Kernel::Line_3         Line_3;


using namespace easy3d;
using namespace cv;


void edit_model(PaintCanvas *viewer_) {
    auto cloud = dynamic_cast<PointCloud*>(viewer_->currentModel());

    if (cloud->n_vertices() >= 1000000) // stop growing when the model is too big
        return;

    auto colors = cloud->vertex_property<vec3>("v:color");
    for (int i = 0; i < 100; ++i) {
        auto v = cloud->add_vertex(vec3(random_float(), random_float(), random_float()));
        colors[v] = vec3(random_float(), random_float(), random_float()); // we use a random color
    }

    cloud->renderer()->update();

    // viewer_->makeCurrent();
    // auto tex_ = viewer_->texture();
    // if (tex_)
    //     delete tex_;
    // tex_ = easy3d::Texture::create("/home/zhl/Downloads/opencv_build/Easy3D-mac/Easy3D-mac/resources/data/caifang/rgba.png");
    // viewer_->doneCurrent();

    viewer_->update();

    // std::cout << "#points: " << cloud->n_vertices() << std::endl;
}

void edit_model_gpu(PaintCanvas *viewer_) {
    const std::vector<vec3> &points = resource::bunny_vertices;
    auto cloud = dynamic_cast<PointCloud*>(viewer_->currentModel());

    auto drawable = cloud->renderer()->get_points_drawable("vertices");
    viewer_->makeCurrent();

    void* pointer = VertexArrayObject::map_buffer(GL_ARRAY_BUFFER, drawable->vertex_buffer(), GL_WRITE_ONLY);

    vec3* vertices = reinterpret_cast<vec3*>(pointer);
    if (!vertices)
        return ;

    static float total_scale = 1.0f;
    float scale = 1.01f;
    if (total_scale > 1.5f) {
        scale = 1.0f / total_scale;
        total_scale = 1.0f;
    }
    else
        total_scale *= scale;

    for (std::size_t i=0; i<points.size(); ++i)
        vertices[i].z *= scale;

    // unmap the vertex buffer
    VertexArrayObject::unmap_buffer(GL_ARRAY_BUFFER, drawable->vertex_buffer());

    viewer_->doneCurrent();    
    viewer_->update();
}


namespace internal {

        // vec3 p0, p1;
        /// When an intersecting point (at an edge, computed from a plane and an edge)
        /// is very close to an existing vertex (i.e., an end point of an edge), we
        /// snap the intersecting point to the existing vertex. This way we can avoid
        /// many thin faces.
        /// \note Value really doesn't matter as long as it is small (default is 1e-10).
        ///       So this parameter is not intended to be changed by the user.
        /// TODO: make this in the API
        float snap_squared_distance_threshold() {
            return float(1e-10);
        }

        void down_sample(PointCloud *cloud, float threshold, int k_least, bool grid) {
            // float threshold = 0.03;
            unsigned int k = 16;
            // unsigned int k_least = 900;  PointCloudSimplification::grid_simplification(cloud, threshold);

            // easy3d::KdTreeSearch*    kdtree_;
            std::vector<easy3d::PointCloud::Vertex> points_to_remove_;

            // kdtree_ = new KdTreeSearch_ETH(cloud);
            // points_to_remove_ = PointCloudSimplification::uniform_simplification(cloud, threshold);
            // Another way
            if (grid)
                points_to_remove_ = PointCloudSimplification::grid_simplification(cloud, threshold);
            else
                points_to_remove_ = PointCloudSimplification::uniform_simplification(cloud, threshold);
            for (auto v : points_to_remove_)
                cloud->delete_vertex(v);
            cloud->collect_garbage();

            // points_to_remove_ = PointCloudSimplification::deisolate(cloud, threshold +0.01, 2);
            // for (auto v : points_to_remove_)
            //     cloud->delete_vertex(v);
            // cloud->collect_garbage();

            // delete kdtree_;

            PointCloudNormals pcn;
            pcn.estimate(cloud, k);
            // pcn.reorient(cloud, k);

            PrimitivesRansac algo;
            algo.add_primitive_type(PrimitivesRansac::PLANE);
            algo.detect(cloud, k_least, 0.005f, 0.02f, 0.8f, 0.001f);
     
        }

        /// \cond SKIP_IN_MANUAL

        template<typename Planar_segment>
        class SegmentSizeIncreasing {
        public:
            SegmentSizeIncreasing() {}

            bool operator()(const Planar_segment *s0, const Planar_segment *s1) const {
                return s0->size() < s1->size();
            }
        };

        template<typename Planar_segment>
        class SegmentSizeDecreasing {
        public:
            SegmentSizeDecreasing() {}

            bool operator()(const Planar_segment *s0, const Planar_segment *s1) const {
                return s0->size() > s1->size();
            }
        };

        template<typename plane>
        class OriginDistanceIncreasing {
        public:
            OriginDistanceIncreasing() {}

            bool operator()(plane s0, plane s1) const {
                vec3 origin(0, 0, 0);
                return s0.plane->squared_distance(origin) < s1.plane->squared_distance(origin);
            }
        };

        template<typename VT>
        void sort_increasing(VT &v1, VT &v2, VT &v3) {
            VT vmin = 0;
            if (v1 < v2 && v1 < v3)
                vmin = v1;
            else if (v2 < v1 && v2 < v3)
                vmin = v2;
            else
                vmin = v3;

            VT vmid = 0;
            if ((v1 > v2 && v1 < v3) || (v1 < v2 && v1 > v3))
                vmid = v1;
            else if ((v2 > v1 && v2 < v3) || (v2 < v1 && v2 > v3))
                vmid = v2;
            else
                vmid = v3;

            VT vmax = 0;
            if (v1 > v2 && v1 > v3)
                vmax = v1;
            else if (v2 > v1 && v2 > v3)
                vmax = v2;
            else
                vmax = v3;

            v1 = vmin;
            v2 = vmid;
            v3 = vmax;
        }
        /// \endcond

        // A group of points (represented by their indices) belonging to a planar segment in a point set.
        class Planar_segment : public std::vector<std::size_t> {
        public:
            // \param point_set the point set that owns this planar segment.
            Planar_segment(const PointCloud *point_set) : cloud_(point_set), supporting_plane_(nullptr) {}

            ~Planar_segment() { 
                if (supporting_plane_) 
                delete supporting_plane_; 
            }

            const PointCloud *cloud() const { return cloud_; }

            // Fits and returns the supporting plane of this planar segment
            Plane3 *fit_supporting_plane();
            SurfaceMesh *borders();
            SurfaceMesh *borders(Plane3 *new_plane);
            SurfaceMesh *borders_plus(float dd1, float dd2, const Plane3 *plane_lidar);
            SurfaceMesh *borders_alpha();
            SurfaceMesh *borders_alpha_plus(float dd1, float dd2, const Plane3 *plane_lidar, float avg_spacing);
            float plane_deviation();
            void  distance_tail(Plane3 *plane_bottom, const Plane3 *plane_origin, float dd1, float &min, float &max, float &average);
            vec3 plane_center() {return plane_center_; }

            // Returns the supporting plane of this planar segment.
            // \note Returned plane is valid only if fit_supporting_plane() has been called.
            Plane3 *supporting_plane() const { return supporting_plane_; }

        private:
            const PointCloud *cloud_;
            Plane3 *supporting_plane_; // The hypothesis generator owns this plane and manages the memory
            vec3 plane_center_;
        };


        // An enriched point set that stores the extracted planar segments
        class EnrichedPointCloud {
        public:
            EnrichedPointCloud(const PointCloud *cloud, PointCloud::VertexProperty<int> plane_indices) : cloud_(cloud) {
                // Gets to know the number of plane from the plane indices
                int max_plane_index = 0;
                for (auto v: cloud->vertices()) {
                    int plane_index = plane_indices[v];
                    if (plane_index > max_plane_index)
                        max_plane_index = plane_index;
                }
                std::size_t num_plane = max_plane_index + 1; // the first one has index 0

                for (std::size_t i = 0; i < num_plane; ++i)
                    planar_segments_.push_back(new Planar_segment(cloud));

                std::size_t idx = 0;
                for (auto v: cloud->vertices()) {
                    int plane_index = plane_indices[v];
                    if (plane_index != -1) {
                        Planar_segment *ps = planar_segments_[plane_index];
                        ps->push_back(idx);
                    }
                    ++idx;
                }
            }

            ~EnrichedPointCloud() {
                for (std::size_t i = 0; i < planar_segments_.size(); ++i)
                    delete planar_segments_[i];
            }

            const PointCloud *cloud() const { return cloud_; }

            std::vector<Planar_segment *> &planar_segments() { return planar_segments_; }

            const std::vector<Planar_segment *> &planar_segments() const { return planar_segments_; }

        private:
            const PointCloud *cloud_;
            std::vector<Planar_segment *> planar_segments_;
        };


        Plane3 *Planar_segment::fit_supporting_plane() {
            const auto &points = cloud_->points();
            PrincipalAxes<3> pca;
            pca.begin();
            for (std::size_t i = 0; i < size(); ++i) {
                std::size_t idx = at(i);
                pca.add(points[idx]);
            }
            pca.end();

            if (supporting_plane_)
                delete supporting_plane_;

            // the normal is the eigen vector corresponding to the smallest eigen value
            supporting_plane_ = new Plane3(pca.center<float>(), pca.axis<float>(2));
            plane_center_ = pca.center<float>();

            // LOG(INFO) << pca.axis<float>(2);
            // std::string file_name = MainWindow::getCurDataDirectory().toStdString() + "/calib.txt";
            // std::ofstream output(file_name.c_str());
            // if (output.fail()) {
            //     LOG(ERROR) << "could not open file: " << file_name;
            //     return false;
            // }
            // output.precision(16);
            // output << pca.center<float>() << pca.axis<float>(2);

            return supporting_plane_;
        }

        SurfaceMesh *Planar_segment::borders() {
            const auto &pts = cloud_->points();
            auto &plane = supporting_plane_;
            auto &center = plane_center_;

            std::vector<vec2> points;

            for (std::size_t i = 0; i < size(); ++i) {
                std::size_t idx = at(i);
                auto &p = pts[idx];
                points.push_back(vec2(plane->to_2d(p)));
            }

            Delaunay2 delaunay;
            delaunay.set_vertices(points);

            SurfaceMesh* mesh = new SurfaceMesh;
            const std::string &name = "Plane_" +std::to_string(points.size()) ;
            mesh->set_name(name);

            for (std::size_t i = 0; i < points.size(); i++) {
                // mesh->add_vertex(vec3(points[i], pts[i].z));
                // std::size_t idx = at(i);
                mesh->add_vertex(vec3(plane->to_3d(points[i])));
            }

            for (unsigned int i = 0; i < delaunay.nb_triangles(); i++) {
                std::vector<SurfaceMesh::Vertex> vts(3);
                for (int j = 0; j < 3; j++) {
                    const int v = delaunay.tri_vertex(i, j);
                    assert(v >= 0);
                    assert(v < points.size());
                    vts[j] = SurfaceMesh::Vertex(v);
                }
                mesh->add_face(vts);
            }

            return mesh;
        }

        SurfaceMesh *Planar_segment::borders(Plane3 *new_plane) {
            const auto &pts = cloud_->points();
            auto &plane = new_plane;
            // auto &center = plane_center_;

            std::vector<vec2> points;

            for (std::size_t i = 0; i < size(); ++i) {
                std::size_t idx = at(i);
                auto &p = pts[idx];
                points.push_back(vec2(plane->to_2d(p)));
            }

            Delaunay2 delaunay;
            delaunay.set_vertices(points);

            SurfaceMesh* mesh = new SurfaceMesh;
            const std::string &name = "Plane_New_" +std::to_string(points.size()) ;
            mesh->set_name(name);

            for (std::size_t i = 0; i < points.size(); i++) {
                // mesh->add_vertex(vec3(points[i], pts[i].z));
                // std::size_t idx = at(i);
                mesh->add_vertex(vec3(plane->to_3d(points[i])));
            }

            for (unsigned int i = 0; i < delaunay.nb_triangles(); i++) {
                std::vector<SurfaceMesh::Vertex> vts(3);
                for (int j = 0; j < 3; j++) {
                    const int v = delaunay.tri_vertex(i, j);
                    assert(v >= 0);
                    assert(v < points.size());
                    vts[j] = SurfaceMesh::Vertex(v);
                }
                mesh->add_face(vts);
            }

            return mesh;
        }

        SurfaceMesh *Planar_segment::borders_plus(float dd1, float dd2, const Plane3 *plane_lidar) {
            // StopWatch w;
            // w.start();

            const auto &pts = cloud_->points();
            auto &plane = supporting_plane_;
            auto &center = plane_center_;

            std::vector<vec2> points;

            float  center_orient;
            bool k1, k2, k3;

            center_orient = plane_lidar->orient(center);

            for (std::size_t i = 0; i < size(); ++i) {
                std::size_t idx = at(i);
                auto &p = pts[idx];
                k1 = plane->squared_distance(p) < dd1*dd1;
                k2 = plane_lidar->squared_distance(p) > dd2*dd2;
                k3 = plane_lidar->orient(p) * center_orient > 0;

                if (k1 && k2 && k3)
                    points.push_back(vec2(plane->to_2d(p)));
            }
            // LOG(INFO) << size() << points.size();

            Delaunay2 delaunay;
            delaunay.set_vertices(points);

            SurfaceMesh* mesh = new SurfaceMesh;
            // const std::string &name = "Plane_" +std::to_string(points.size()) + "_plus";
            const std::string &name = "Plane_plus";
            mesh->set_name(name);

            for (std::size_t i = 0; i < points.size(); i++) {
                // mesh->add_vertex(vec3(points[i], pts[i].z));
                // std::size_t idx = at(i);
                mesh->add_vertex(vec3(plane->to_3d(points[i])));
            }

            for (unsigned int i = 0; i < delaunay.nb_triangles(); i++) {
                std::vector<SurfaceMesh::Vertex> vts(3);
                for (int j = 0; j < 3; j++) {
                    const int v = delaunay.tri_vertex(i, j);
                    assert(v >= 0);
                    assert(v < points.size());
                    vts[j] = SurfaceMesh::Vertex(v);
                }
                mesh->add_face(vts);
            }

            return mesh;
        }

        float Planar_segment::plane_deviation() {
            // StopWatch w;
            // w.start();
            float dist_max = 0;


            const auto &pts = cloud_->points();
            auto &plane = supporting_plane_;
            auto &center = plane_center_;

            for (std::size_t i = 0; i < size(); ++i) {
                std::size_t idx = at(i);
                auto &p = pts[idx];

                const float sdist = plane->squared_distance(p);
                const float dist = std::sqrt(sdist);
                if (dist > dist_max)
                    dist_max = dist;
    
            }

            return dist_max;
        }

        void  Planar_segment::distance_tail(Plane3 *plane_bottom, const Plane3 *plane_origin, float dd1, 
                                                float &min, float &max, float &average) {
            const auto &pts = cloud_->points();
            auto &center = plane_center_;
            std::vector<float> distances_;
            std::vector<float> sub_distances_;
            std::vector<float> subb_distances_;
            bool k1, k2, k3;
            float sdist;
            double sum;

            float center_orient = plane_bottom->orient(center);

            for (std::size_t i = 0; i < size(); ++i) {
                std::size_t idx = at(i);
                auto &p = pts[idx];
                k1 = plane_bottom->squared_distance(p) > 0.1*0.1;
                k2 = plane_bottom->squared_distance(p) < dd1*dd1;
                k3 = plane_bottom->orient(p) * center_orient > 0;

                if (k1 && k2 && k3){
                    sdist = plane_origin->squared_distance(p);
                    distances_.push_back(sqrt(sdist));
                }
            }
            //
            // std::vector<std::size_t> points_indices;
            // points_indices.insert(points_indices.end(), s1->begin(), s1->end());
            // points_indices.insert(points_indices.end(), s2->begin(), s2->end());

            // Planar_segment *s = new Planar_segment(point_set_->cloud());
            // s->insert(s->end(), points_indices.begin(), points_indices.end());
            // s->fit_supporting_plane();
       
            // new way
            int num = distances_.size() * 0.2f;
            std::sort(distances_.begin(), distances_.end());
            if (distances_.size() > num) {
                sub_distances_ = easy3d::slice(distances_, 0, num -1);
                sum = std::accumulate(sub_distances_.begin(), sub_distances_.end(), 0.0);
                max = sum / num;
                subb_distances_ = easy3d::slice(distances_, distances_.size() -num, distances_.size() -1);
                sum = std::accumulate(subb_distances_.begin(), subb_distances_.end(), 0.0);
                min = sum / num;
            } else {
                std::vector<float>::iterator max_ = std::min_element(distances_.begin(), distances_.end());
                max = *max_;
                std::vector<float>::iterator min_ = std::max_element(distances_.begin(), distances_.end());
                min = *min_;
            }

            if (distances_.size() > 0) {
                sum = std::accumulate(distances_.begin(), distances_.end(), 0.0);
                average = sum / distances_.size();
            }
        }


        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

        /* A vertex class with an additional member representing its index */
        template < class Gt, class VB = CGAL::Triangulation_hierarchy_vertex_base_2<Gt> >
        class AS_vertex_base : public VB
        {
        public:
            typedef VB                                                                Base;
            typedef typename VB::Vertex_handle      Vertex_handle;
            typedef typename VB::Face_handle        Face_handle;
            typedef typename VB::Point              Point;

            template < typename TDS2 >
            struct Rebind_TDS {
                typedef typename VB::template Rebind_TDS<TDS2>::Other        VB2;
                typedef AS_vertex_base<Gt, VB2>                         Other;
            };

        public:
            AS_vertex_base() : Base(), index_(-1) {}
            AS_vertex_base(const Point & p) : Base(p), index_(-1) {}
            AS_vertex_base(const Point & p, Face_handle f) : Base(f, p), index_(-1) {}
            AS_vertex_base(Face_handle f) : Base(f), index_(-1) {}

            void set_index(int idx) { index_ = idx; }
            int  index() const { return index_; }

        private:
            int index_;
        };


        template <typename Ht>
        class Alpha_shape : public CGAL::Alpha_shape_2<Ht>
        {
        public:
            typedef CGAL::Alpha_shape_2<Ht>             Parent_class;
            typedef typename Kernel::Point_2                Point2;
            typedef typename Parent_class::Vertex_handle    Vertex_handle;

        public:
            // constructs alpha shapes from the input points
            template <typename InputIterator>
            Alpha_shape(InputIterator first, InputIterator beyond);
        };


        // An Alpha Shape Mesh approximates the point covered region by a mesh representation.
        class Alpha_shape_mesh
        {
            typedef typename Kernel::FT             FT;
            typedef typename Kernel::Point_2        Point2;
            typedef typename Kernel::Plane_3        Plane_3;
            // typedef typename easy3d::Plane3         Plane_3;

            typedef CGAL::Alpha_shape_vertex_base_2<Kernel>     Avb;
            typedef AS_vertex_base<Avb>                         Av;
            typedef CGAL::Triangulation_face_base_2<Kernel>     Tf;
            typedef CGAL::Alpha_shape_face_base_2<Kernel, Tf>   Af;
            typedef CGAL::Triangulation_default_data_structure_2<Kernel, Av, Af> Tds;
            typedef CGAL::Delaunay_triangulation_2<Kernel, Tds> Dt;
            typedef CGAL::Triangulation_hierarchy_2<Dt>         Ht;

        public:
            /// Given a set of 3D points lying on 'plane', constructs alpha shapes from the
            /// the projection of the points onto 'plane'
            template <typename InputIterator>
            Alpha_shape_mesh(InputIterator first, InputIterator beyond, const Plane_3& plane);

            ~Alpha_shape_mesh() { delete alpha_shape_; }

            /// Extracts the 3D mesh representation of the alpha shapes
            bool extract_mesh(FT alpha_value, SurfaceMesh& mesh);

        private:
            Alpha_shape<Ht>*            alpha_shape_;
            std::vector<const Point_3*>  original_points_;
        };

        template <typename Ht>
        template <typename InputIterator>
        Alpha_shape<Ht>::Alpha_shape(InputIterator first, InputIterator beyond) {
            InputIterator it = first;
            for (int id = 0; it != beyond; ++it, ++id) {
                const Point2& p = *it;
                Vertex_handle vh = Ht::insert(p);
                if (vh->index() == -1)
                    vh->set_index(id);
                else {
                    // p was not inserted (there might be a duplicated point)
                }
            }

            if (Parent_class::dimension() == 2) {
                // Computes the associated _interval_face_map
                Parent_class::initialize_interval_face_map();

                // Computes the associated _interval_edge_map
                Parent_class::initialize_interval_edge_map();

                // Computes the associated _interval_vertex_map
                Parent_class::initialize_interval_vertex_map();

                // Merges the two maps
                Parent_class::initialize_alpha_spectrum();
            }
        }


        template <typename InputIterator>
        Alpha_shape_mesh::Alpha_shape_mesh(InputIterator first, InputIterator beyond, const Plane_3& plane) {
            original_points_.clear();

            std::vector<Point2> pts;
            for (InputIterator it = first; it != beyond; ++it) {
                const Point_3& p = *it;
                const Point2& q = plane.to_2d(p);
                pts.push_back(q);
                original_points_.push_back(&p);
            }
            alpha_shape_ = new Alpha_shape<Ht>(pts.begin(), pts.end());
        }


        bool Alpha_shape_mesh::extract_mesh(FT alpha_value, SurfaceMesh& mesh) {
            alpha_shape_->set_alpha(alpha_value);

            typedef std::vector<std::size_t> Triangle;
            std::vector<Triangle>        faces;

            typedef Alpha_shape<Ht>        Alpha_shape;

            typename Alpha_shape::Finite_faces_iterator fit = alpha_shape_->finite_faces_begin();
            for (; fit != alpha_shape_->finite_faces_end(); ++fit) {
                if (alpha_shape_->classify(fit) == Alpha_shape::INTERIOR) {
                    Triangle tri;
                    for (int i = 0; i < 3; ++i) {
                        typename Alpha_shape::Vertex_handle vh = fit->vertex(i);
                        int idx = vh->index();
                        tri.push_back(idx);
                    }
                    faces.push_back(tri);
                }
            }

            if (faces.empty())
                return false;

            mesh.clear();

            std::vector<SurfaceMesh::Vertex> descriptors(original_points_.size());
            for (std::size_t i = 0; i < original_points_.size(); ++i) {
                const Point_3* p = original_points_[i];
                descriptors[i] = mesh.add_vertex(vec3(p->x(), p->y(), p->z()));
            }

            for (std::size_t i = 0; i < faces.size(); ++i) {
                std::vector<SurfaceMesh::Vertex> face;
                const Triangle& tri = faces[i];
                for (std::size_t j = 0; j < tri.size(); ++j) {
                    std::size_t idx = tri[j];
                    face.push_back(descriptors[idx]);
                }
                mesh.add_face(face);;
            }

            return true;
        }

        SurfaceMesh *Planar_segment::borders_alpha() {
            const auto &pts = cloud_->points();
            auto &plane = supporting_plane_;
            const vec3 &pp = plane->point();
            const vec3 &ppn = plane->normal();
            const Plane_3* supporting_plane = new Plane_3(Point_3(pp.x, pp.y, pp.z), Vector_3(ppn.x, ppn.y, ppn.z));

            std::vector<Point_3> points;

            for (std::size_t i = 0; i < size(); ++i) {
                std::size_t idx = at(i);
                const vec3 &p = pts[idx];

                if (plane->squared_distance(p) < 0.06*0.06) {
                    vec3 p_temp = plane->projection(p);
                    points.push_back(Point_3(p_temp.x, p_temp.y, p_temp.z));
                }
            }

            float avg_spacing = 0.03;
            Alpha_shape_mesh alpha_mesh(points.begin(), points.end(), *supporting_plane);

            SurfaceMesh* covering_mesh = new SurfaceMesh;
            const std::string &name = "Plane_" +std::to_string(points.size()) + "_alpha_shape";
            covering_mesh->set_name(name);

            float radius = avg_spacing * float(5.0);
            if (alpha_mesh.extract_mesh(radius * radius, *covering_mesh))
                return covering_mesh ;
            else {
                delete covering_mesh;
                return nullptr ;
            }
        }

        SurfaceMesh *Planar_segment::borders_alpha_plus(float dd1, float dd2, const Plane3 *plane_lidar, float avg_spacing) {
            const auto &pts = cloud_->points();
            auto &plane = supporting_plane_;
            const vec3 &pp = plane->point();
            const vec3 &ppn = plane->normal();
            const Plane_3* supporting_plane = new Plane_3(Point_3(pp.x, pp.y, pp.z), Vector_3(ppn.x, ppn.y, ppn.z));
            auto &center = plane_center_;

            std::vector<Point_3> points;
            float  center_orient;
            bool k1, k2, k3;

            center_orient = plane_lidar->orient(center);

            for (std::size_t i = 0; i < size(); ++i) {
                std::size_t idx = at(i);
                const vec3 &p = pts[idx];

                k1 = plane->squared_distance(p) < dd1*dd1;
                k2 = plane_lidar->squared_distance(p) > dd2*dd2;
                k3 = plane_lidar->orient(p) * center_orient > 0;
                if (k1 && k2 && k3) {
                    vec3 p_temp = plane->projection(p);
                    points.push_back(Point_3(p_temp.x, p_temp.y, p_temp.z));
                }
            }

            // float avg_spacing = 0.03;
            Alpha_shape_mesh alpha_mesh(points.begin(), points.end(), *supporting_plane);

            SurfaceMesh* covering_mesh = new SurfaceMesh;
            // const std::string &name = "Plane_" +std::to_string(points.size()) + "_alpha_shape";
            const std::string &name = "Plane_plus";
            covering_mesh->set_name(name);

            float radius = avg_spacing * float(5.0);
            if (alpha_mesh.extract_mesh(radius * radius, *covering_mesh))
                return covering_mesh ;
            else {
                delete covering_mesh;
                return nullptr ;
            }
        }
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

        class Hypothesis {
        public:
            Hypothesis() {}

            ~Hypothesis();

            void generate(EnrichedPointCloud &point_set);
            std::vector<Plane3 *> &supporting_planes() { return supporting_planes_; }
            std::vector<SurfaceMesh *> &supporting_meshes() { return supporting_meshes_; }
            std::vector<vec3> &supporting_planes_center() { return supporting_planes_center_; }
            std::vector<Planar_segment *> &supporting_segments() { return point_set_->planar_segments(); }
        public:
            void refine_plane_sub(Planar_segment *s, Plane3 *plane_bottom, const Plane3 *plane_origin, float dd1, 
                                                float &min, float &max, float &average);

        private:
            // Merges near co-planar segments
            void refine_planes();
            void clear();
            // Counts the number of points that are with the dist_threshold to its supporting plane
            std::size_t number_of_points_on_plane(const Planar_segment *s, const Plane3 *plane, float dist_threshold);
            void merge(Planar_segment *s1, Planar_segment *s2);

        private:
            // The input point cloud with planes
            EnrichedPointCloud *point_set_;
            std::vector<Plane3 *> supporting_planes_;
            std::vector<SurfaceMesh *> supporting_meshes_;
            std::vector<vec3> supporting_planes_center_;

        }; // end Hypothesis

        Hypothesis::~Hypothesis() {
            clear();
        }


        void Hypothesis::clear() {
            // for (std::size_t i = 0; i < supporting_planes_.size(); ++i)
            //     delete supporting_planes_[i];
            supporting_planes_.clear();

            for (std::size_t i = 0; i < supporting_meshes_.size(); ++i)
                delete supporting_meshes_[i];
            supporting_meshes_.clear();

        }

        std::size_t
        Hypothesis::number_of_points_on_plane(const Planar_segment *s, const Plane3 *plane, float dist_threshold) {
            assert(const_cast<Planar_segment *>(s)->cloud() == point_set_->cloud());

            std::size_t count = 0;
            const auto &points = point_set_->cloud()->points();
            for (std::size_t i = 0; i < s->size(); ++i) {
                std::size_t idx = s->at(i);
                const vec3 &p = points[idx];

                const float sdist = plane->squared_distance(p);
                const float dist = std::sqrt(sdist);
                if (dist < dist_threshold)
                    ++count;
            }
            return count;
        }

        void Hypothesis::merge(Planar_segment *s1, Planar_segment *s2) {
            assert(const_cast<Planar_segment *>(s1)->cloud() == point_set_->cloud());
            assert(const_cast<Planar_segment *>(s2)->cloud() == point_set_->cloud());
            std::vector<Planar_segment *> &segments = point_set_->planar_segments();

            std::vector<std::size_t> points_indices;
            points_indices.insert(points_indices.end(), s1->begin(), s1->end());
            points_indices.insert(points_indices.end(), s2->begin(), s2->end());

            Planar_segment *s = new Planar_segment(point_set_->cloud());
            s->insert(s->end(), points_indices.begin(), points_indices.end());
            s->fit_supporting_plane();
            segments.push_back(s);

            typename std::vector<Planar_segment *>::iterator pos = std::find(segments.begin(), segments.end(), s1);
            if (pos != segments.end()) {
                Planar_segment *tmp = *pos;
                // const Plane3 *plane = tmp->supporting_plane();
                segments.erase(pos);
                delete tmp;
                // delete plane;
            } else
                std::cerr << "Fatal error: should not reach here" << std::endl;

            pos = std::find(segments.begin(), segments.end(), s2);
            if (pos != segments.end()) {
                Planar_segment *tmp = *pos;
                // const Plane3 *plane = tmp->supporting_plane();
                segments.erase(pos);
                delete tmp;
                // delete plane;
            } else
                std::cerr << "Fatal error: should not reach here" << std::endl;
        }

        void Hypothesis::refine_plane_sub(Planar_segment *s, Plane3 *plane_bottom, const Plane3 *plane_origin, float dd1, 
                                                float &min, float &max, float &average) {
            const auto &pts = point_set_->cloud()->points();
            auto center = s->plane_center();
            std::vector<float> distances_;
            std::vector<float> sub_distances_;
            std::vector<float> subb_distances_;
            bool k1, k2, k3;
            float sdist;
            double sum;
            std::vector<std::size_t> points_indices;

            float center_orient = plane_bottom->orient(center);

            for (std::size_t i = 0; i < s->size(); ++i) {
                std::size_t idx = s->at(i);
                auto &p = pts[idx];
                k1 = plane_bottom->squared_distance(p) > 0.03*0.03;
                k2 = plane_bottom->squared_distance(p) < dd1*dd1;
                k3 = plane_bottom->orient(p) * center_orient > 0;

                if (k1 && k2 && k3){
                    points_indices.push_back(idx);
                }
            }

            Planar_segment *s_sub = new Planar_segment(point_set_->cloud());
            s_sub->insert(s_sub->end(), points_indices.begin(), points_indices.end());
            s_sub->fit_supporting_plane();

            for (std::size_t i = 0; i < s_sub->size(); ++i) {
                std::size_t idx = s_sub->at(i);
                auto &p = pts[idx];
                sdist = plane_origin->squared_distance(s_sub->supporting_plane()->projection(p));
                distances_.push_back(sqrt(sdist));
            }

            // new way
            int num = distances_.size() * 0.2f;
            std::sort(distances_.begin(), distances_.end());
            if (distances_.size() > num) {
                sub_distances_ = easy3d::slice(distances_, 0, num -1);
                sum = std::accumulate(sub_distances_.begin(), sub_distances_.end(), 0.0);
                max = sum / num;
                subb_distances_ = easy3d::slice(distances_, distances_.size() -num, distances_.size() -1);
                sum = std::accumulate(subb_distances_.begin(), subb_distances_.end(), 0.0);
                min = sum / num;
            } else {
                std::vector<float>::iterator max_ = std::min_element(distances_.begin(), distances_.end());
                max = *max_;
                std::vector<float>::iterator min_ = std::max_element(distances_.begin(), distances_.end());
                min = *min_;
            }

            // if (distances_.size() > 0) {
            //     sum = std::accumulate(distances_.begin(), distances_.end(), 0.0);
            //     average = sum / distances_.size();
            // }
            sdist = plane_origin->squared_distance(s_sub->plane_center());
            average = sqrt(sdist);

            delete s_sub;

            return;
        }

        void Hypothesis::refine_planes() {
            std::vector<Planar_segment *> &segments = point_set_->planar_segments();
            const auto &points = point_set_->cloud()->points();

            float avg_max_dist = 0;
            for (std::size_t i = 0; i < segments.size(); ++i) {
                Planar_segment *s = segments[i];
                const Plane3 *plane = s->fit_supporting_plane(); // user may provide invalid plane fitting (we always fit)

                float max_dist = -(std::numeric_limits<float>::max)();
                for (std::size_t j = 0; j < s->size(); ++j) {
                    std::size_t idx = s->at(j);
                    const vec3 &p = points[idx];
                    const float sdist = plane->squared_distance(p);
                    max_dist = (std::max)(max_dist, std::sqrt(sdist));
                }

                avg_max_dist += max_dist;
            }
            avg_max_dist /= segments.size();
            avg_max_dist /= 2.0f;

            const float theta = geom::to_radians(10.0);        
            
            bool merged = false;
            do {
                merged = false;
                // Segments with less points have less confidences and thus should be merged first.
                // So we sort the segments according to their sizes.
                std::sort(segments.begin(), segments.end(), ::internal::SegmentSizeIncreasing<Planar_segment>());
                    // LOG(INFO) << "enter while...";
                for (std::size_t i = 0; i < segments.size(); ++i) {
                    Planar_segment *s1 = segments[i];
                    const Plane3 *plane1 = s1->supporting_plane();
                    const vec3 n1 = plane1->normal().normalize();

                    const float num_threshold = s1->size() / 5.0f;
                    for (std::size_t j = i + 1; j < segments.size(); ++j) {
                        Planar_segment *s2 = segments[j];
                        const Plane3 *plane2 = s2->supporting_plane();
                        const vec3 n2 = plane2->normal().normalize();

                        if (std::abs(dot(n1, n2)) > std::cos(theta)) {
                            std::size_t set1on2 = number_of_points_on_plane(s1, plane2, avg_max_dist);
                            std::size_t set2on1 = number_of_points_on_plane(s2, plane1, avg_max_dist);
                            if (set1on2 > num_threshold || set2on1 > num_threshold) {
                                merge(s1, s2);
                                merged = true;
                                break;
                            }
                        }
                    }
                    if (merged)
                        break;
                }
            } while (merged);

            std::sort(segments.begin(), segments.end(), ::internal::SegmentSizeDecreasing<Planar_segment>());

            // Stores all the supporting planes
            for (std::size_t i = 0; i < segments.size(); ++i) {
                Planar_segment *s = segments[i];

                s->fit_supporting_plane();
                Plane3 *plane = s->supporting_plane();
                supporting_planes_.push_back(plane);

                // SurfaceMesh *mesh = s->borders_alpha();
                // SurfaceMesh *mesh = s->borders();
                // supporting_meshes_.push_back(mesh);

                supporting_planes_center_.push_back(s->plane_center());
            }
        }

        void Hypothesis::generate(EnrichedPointCloud &point_set) {
            point_set_ = &point_set;
            refine_planes();
        }

} //end name space

struct Plane_Center {
    Plane3 * plane;
    vec3     center;
    SurfaceMesh * mesh;
    ::internal::Planar_segment *seg;
} ;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->treeWidgetModels->init(this);

    viewer_ = new PaintCanvas(this);
    setCentralWidget(viewer_);

    // ----- the width of the rendering panel ------
    // sizeHint() doesn't suggest a good value
    // const QSize& size = ui->dockWidgetRendering->sizeHint();
    const int width = 270; //static_cast<int>(size.width() * 1.5f);
    ui->dockWidgetRendering->setFixedWidth(width);

    // ------ rendering panel ------

    widgetTrianglesDrawable_ = new WidgetTrianglesDrawable(this);
    ui->verticalLayoutTrianglesDrawable->addWidget(widgetTrianglesDrawable_);
    widgetTrianglesDrawable_->setEnabled(false);

    widgetLinesDrawable_ = new WidgetLinesDrawable(this);
    ui->verticalLayoutLinesDrawable->addWidget(widgetLinesDrawable_);
    widgetLinesDrawable_->setEnabled(false);

    widgetPointsDrawable_ = new WidgetPointsDrawable(this);
    ui->verticalLayoutPointsDrawable->addWidget(widgetPointsDrawable_);
    widgetPointsDrawable_->setEnabled(false);

    auto widgetGlobalSetting = new WidgetGlobalSetting(this);
    ui->verticalLayoutGlobalSetting->addWidget(widgetGlobalSetting);

    // communication between widgets
    widgetGlobalSetting->widgetTrianglesDrawable_ = widgetTrianglesDrawable_;

    // ---------------------------

    // file menu
    createActionsForFileMenu();

    // view menu
    createActionsForViewMenu();

    // view menu
    createActionsForCameraMenu();

    // edit menu
    createActionsForEditMenu();

    // property menu
    createActionsForPropertyMenu();

    // select menu
    createActionsForSelectMenu();

    // point cloud menu
    createActionsForPointCloudMenu();

    // surface mesh menu
    createActionsForSurfaceMeshMenu();

    // polyhedral mesh menu
    createActionsForPolyMeshMenu();

    // status bar
    createStatusBar();

    // about menu
    connect(ui->actionAbout, SIGNAL(triggered()), this, SLOT(onAbout()));
    connect(ui->actionManual, SIGNAL(triggered()), this, SLOT(showManual()));
    connect(ui->actionboundary, SIGNAL(triggered()), this, SLOT(show_boundary()));
    connect(ui->actionbbox, SIGNAL(triggered()), this, SLOT(show_bbox()));
    connect(ui->actionbox, SIGNAL(triggered()), this, SLOT(show_box()));
    connect(ui->actiononekey, SIGNAL(triggered()), this, SLOT(show_onekey()));
    connect(ui->actionSave_Plane_Param, SIGNAL(triggered()), this, SLOT(Save_Plane_Param()));

    // options for the model panel
    connect(ui->checkBoxAutoFocus, SIGNAL(toggled(bool)), ui->treeWidgetModels, SLOT(setAutoFocus(bool)));
    connect(ui->checkBoxSelectedOnly, SIGNAL(toggled(bool)), ui->treeWidgetModels, SLOT(setSelectedOnly(bool)));

    // connect test
    connect(this, SIGNAL(test_vec3(easy3d::vec3 a)), this, SLOT(show_vec3(easy3d::vec3 a)));

    ui->mainToolBar->addSeparator();

#if 1
    auto actionTest = ui->mainToolBar->addAction("Test");
    connect(actionTest, &QAction::triggered, [this]() -> void {
        Model* m = viewer_->currentModel();
        PointCloud *cloud = dynamic_cast<PointCloud *>(viewer_->currentModel());
        if (cloud) {
            // ::internal::down_sample(cloud);
            std::vector<easy3d::PointCloud::Vertex> points_to_remove_;
            points_to_remove_ = PointCloudSimplification::deisolate(cloud, 0.03 +0.01, 3);
            for (auto v : points_to_remove_)
                cloud->delete_vertex(v);
            cloud->collect_garbage();

            cloud->renderer()->update();
            viewer_->update();
        }
        // if (!m)
        //     return;

        // viewer_->makeCurrent();
        // viewer_->doneCurrent();
        // viewer_->update();
        // updateUi();

        
    // edit_model_gpu(viewer());

        


    });
#endif

#if 1
    ui->mainToolBar->addSeparator();
    auto actionYAML = ui->mainToolBar->addAction("rotate_line_calculate");
    connect(actionYAML, &QAction::triggered, [this]() -> void {

        read_calibrated_rt(".");
        
    });
#endif

    setWindowIcon(QIcon(QString::fromStdString(":/resources/icons/Mapple.png")));
    setContextMenuPolicy(Qt::CustomContextMenu);
    setAcceptDrops(true);

#ifdef NDEBUG
    setWindowState(Qt::WindowMaximized);
#else
    setBaseSize(1024, 800);
#endif

    readSettings();
    updateWindowTitle();
}


MainWindow::~MainWindow() {
    LOG(INFO) << "Mapple terminated. Bye!";
}

void MainWindow::show_vec3(easy3d::vec3 a) {
    LOG(INFO) << a;
}

bool MainWindow::read_calibrated_rt(const std::string& dir) {

    Yaml::Node params_;
    Yaml::Parse (params_, "./params.yaml");
    vec3 plane_center, plane_normal;
    std::string plane_ground_s = params_["plane_front_up"].As<std::string> ();
    std::stringstream plane_ground_stream(plane_ground_s);
    plane_ground_stream >> plane_center >> plane_normal;

    const std::string cam_dir = dir + "/rotate_line";
    std::vector<std::string> cam_files;
    std::vector<Plane3 *> planes;
    file_system::get_directory_entries(cam_dir, cam_files, false);
    std::sort(cam_files.begin(), cam_files.end());

    bool ok;
    Line3 line;

    for (const auto& entry : cam_files) {
        const std::string file_name = cam_dir + "/" + entry;
        std::ifstream input(file_name.c_str());
        if (input.fail()) {
            std::cerr << "could not open file \'" << file_name << "\'" << std::endl;
            continue;
        }
        vec3 p, n;        
        input >> p >> n;

        auto plane_temp = new Plane3(p, n);
        planes.push_back(plane_temp);  
    }

    ok = planes[0]->intersect(*planes[1], line);
    if (ok) {
        LOG(INFO) << "ok: " << line;
        float thetra = geom::to_degrees(geom::angle2(planes[0]->normal(), planes[1]->normal()));
        LOG(INFO) << "rotation_two_plane_normal_deviation: " << thetra;
        thetra = geom::to_degrees(geom::angle2(line.direction(), plane_normal));
        LOG(INFO) << "rotation_line_ground_normal_deviation: " << thetra;
    }

    for ( auto &p : planes) {
        // LOG(INFO) << *p;
        delete p;
    }

    return cam_files.size() > 0;
}

void MainWindow::notify(std::size_t percent, bool update_viewer) {
    progress_bar_->setValue(int(percent));
    cancelTaskButton_->setVisible(percent > 0 && percent < 100);
    progress_bar_->setVisible(percent > 0 && percent < 100);

    if (update_viewer)
        viewer_->update();

    // Force updating UI.
    // This approach has significant drawbacks. For example, imagine you wanted to perform two such loops
    // in parallel-calling one of them would effectively halt the other until the first one is finished
    // (so you can't distribute computing power among different tasks). It also makes the application react
    // with delays to events. Furthermore, the code is difficult to read and analyze, therefore this solution
    // is only suited for short and simple problems that are to be processed in a single thread, such as
    // splash screens and the monitoring of short operations.
    QApplication::processEvents();
}


void MainWindow::send(el::Level level, const std::string &msg) {
    static QMutex mutex;
    mutex.lock();
    std::string line("");
	switch (level) {
        case el::Level::Info:
            ui->listWidgetLog->addItem(QString::fromStdString("[INFO] " + msg));
            // set to black will not work if it is dark mode
//            ui->listWidgetLog->item(ui->listWidgetLog->count() - 1)->setForeground(Qt::black);
            break;
        case el::Level::Warning:
            ui->listWidgetLog->addItem(QString::fromStdString("[WARNING] " + msg));
            ui->listWidgetLog->item(ui->listWidgetLog->count() - 1)->setForeground(Qt::darkRed);
            break;
        case el::Level::Error:
            ui->listWidgetLog->addItem(QString::fromStdString("[ERROR] " + msg));
            ui->listWidgetLog->item(ui->listWidgetLog->count() - 1)->setForeground(Qt::red);
            break;
        case el::Level::Fatal:  // no need to handle (app will crash)
        default: break;
    }

    ui->listWidgetLog->scrollToBottom();
	mutex.unlock();
}


void MainWindow::createStatusBar()
{
    labelStatusInfo_ = new QLabel("Ready", this);
    labelStatusInfo_->setFixedWidth(static_cast<int>(ui->dockWidgetRendering->width() * 2.0f));
    labelStatusInfo_->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    statusBar()->addWidget(labelStatusInfo_);

    //////////////////////////////////////////////////////////////////////////

    const int length = 100;
    labelNumFaces_ = new QLabel(this);
    labelNumFaces_->setMinimumWidth(length);
    labelNumFaces_->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    statusBar()->addWidget(labelNumFaces_);

    labelNumVertices_ = new QLabel(this);
    labelNumVertices_->setMinimumWidth(length);
    labelNumVertices_->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    statusBar()->addWidget(labelNumVertices_);

    labelNumEdges_ = new QLabel(this);
    labelNumEdges_->setMinimumWidth(length);
    labelNumEdges_->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    statusBar()->addWidget(labelNumEdges_);

    labelNumCells_ = new QLabel(this);
    labelNumCells_->setMinimumWidth(length);
    labelNumCells_->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    statusBar()->addWidget(labelNumCells_);

    //////////////////////////////////////////////////////////////////////////

    cancelTaskButton_ = new QPushButton(this);
    cancelTaskButton_->setVisible(false);
    cancelTaskButton_->setFlat(true);
    cancelTaskButton_->setIcon(QIcon(":/resources/icons/cancel.png"));
    cancelTaskButton_->setFixedSize(30, 30);
    statusBar()->addPermanentWidget(cancelTaskButton_, 1);
    connect(cancelTaskButton_, SIGNAL(pressed()), this,  SLOT(cancelTask()));

    progress_bar_ = new QProgressBar(this);
    progress_bar_->setVisible(false);
    progress_bar_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    progress_bar_->setFixedWidth(ui->dockWidgetModels->width());
    statusBar()->addPermanentWidget(progress_bar_);

    //////////////////////////////////////////////////////////////////////////

    updateStatusBar();
}


void MainWindow::updateStatusBar()
{
    QString faces(""), vertices(""), edges(""), cells("");

    Model* model = viewer_->currentModel();
    if (dynamic_cast<SurfaceMesh*>(model)) {
        auto mesh = dynamic_cast<SurfaceMesh*>(model);
        faces = QString("#faces: %1  ").arg(mesh->n_faces());
        vertices = QString("#vertices: %1  ").arg(mesh->n_vertices());
        edges = QString("#edges: %1  ").arg(mesh->n_edges());
        labelNumFaces_->setVisible(true);
        labelNumEdges_->setVisible(true);
        labelNumCells_->setVisible(false);
    }

    else if (dynamic_cast<PointCloud*>(model)) {
        auto cloud = dynamic_cast<PointCloud*>(model);
        vertices = QString("#vertices: %1  ").arg(cloud->n_vertices());
        labelNumFaces_->setVisible(false);
        labelNumEdges_->setVisible(false);
        labelNumCells_->setVisible(false);
    }

    else if (dynamic_cast<Graph*>(model)) {
        auto graph = dynamic_cast<Graph*>(model);
        vertices = QString("#vertices: %1  ").arg(graph->n_vertices());
        edges = QString("#edges: %1  ").arg(graph->n_edges());
        labelNumFaces_->setVisible(false);
        labelNumEdges_->setVisible(true);
        labelNumCells_->setVisible(false);
    }

    else if (dynamic_cast<PolyMesh*>(model)) {
        auto mesh = dynamic_cast<PolyMesh*>(model);
        faces = QString("#faces: %1  ").arg(mesh->n_faces());
        vertices = QString("#vertices: %1  ").arg(mesh->n_vertices());
        edges = QString("#edges: %1  ").arg(mesh->n_edges());
        cells = QString("#cells: %1  ").arg(mesh->n_cells());
        labelNumFaces_->setVisible(true);
        labelNumEdges_->setVisible(true);
        labelNumCells_->setVisible(true);
    }


    labelNumVertices_->setText( vertices );
    labelNumFaces_->setText( faces );
    labelNumCells_->setText( cells );
    labelNumEdges_->setText( edges );
}


void MainWindow::cancelTask() {
    cancel();
    cancelTaskButton_->setVisible(false);
    progress_bar_->reset();
    progress_bar_->setTextVisible(false);
    progress_bar_->setVisible(false);
    viewer_->update();
}


void MainWindow::dragEnterEvent(QDragEnterEvent *e) {
    if (e->mimeData()->hasUrls())
        e->acceptProposedAction();
}


void MainWindow::dropEvent(QDropEvent *e) {
    if (e->mimeData()->hasUrls())
        e->acceptProposedAction();

    int count = 0;
    foreach (const QUrl &url, e->mimeData()->urls()) {
        const QString &fileName = url.toLocalFile();
        if (open(fileName.toStdString()))
            ++count;
    }

    if (count > 0)
        viewer_->update();
}


int MainWindow::openFiles(const QStringList &fileNames) {
    int count = 0;
    ProgressLogger progress(fileNames.size(), true, false);
    for (const auto& name : fileNames) {
        if (progress.is_canceled()) {
            LOG(WARNING) << "opening files cancelled";
            break;
        }
        if (open(name.toStdString()))
            ++count;
        progress.next();
    }

    return count > 0;
}


void MainWindow::loadModelTranslateChanged(QAction* act) {
    if (act == ui->actionTranslateDisabled) {
        Translator::instance()->set_status(Translator::DISABLED);
        LOG(INFO) << "translation in file IO has been disabled";
    } else if (act == ui->actionTranslateUseFirstVertex) {
        Translator::instance()->set_status(Translator::TRANSLATE_USE_FIRST_POINT);
        LOG(INFO) << "translation with respect to first vertex in file IO";
    }
    else if (act == ui->actionTranslateUseLastKnownVertex) {
        Translator::instance()->set_status(Translator::TRANSLATE_USE_LAST_KNOWN_OFFSET);
        const dvec3 &origin = Translator::instance()->translation();
        LOG(INFO) << "translation with respect to last know vertex (" << origin << ") in file IO";
    }
}


bool MainWindow::onOpen() {
    const QStringList& fileNames = QFileDialog::getOpenFileNames(
                this,
                "Open file(s)",
                curDataDirectory_,
                "Supported formats (*.ply *.obj *.off *.stl *.sm *.geojson *.trilist *.bin *.las *.laz *.xyz *.bxyz *.vg *.bvg *.ptx *.plm *.pm *.mesh)\n"
                "Surface Mesh (*.ply *.obj *.off *.stl *.sm *.geojson *.trilist)\n"
                "Point Cloud (*.ply *.bin *.ptx *.las *.laz *.xyz *.bxyz *.vg *.bvg *.ptx)\n"
                "Polyhedral Mesh (*.plm *.pm *.mesh)\n"
                "Graph (*.ply)\n"
                "All formats (*.*)"
            );

    // Hide closed dialog
    QApplication::processEvents();

    if (fileNames.empty())
        return false;

    return openFiles(fileNames);
}


bool MainWindow::onSave() {
    const Model* model = viewer_->currentModel();
    if (!model) {
        std::cerr << "no model exists" << std::endl;
        return false;
    }

    std::string default_file_name = model->name();
    if (file_system::extension(default_file_name).empty()) // no extension?
        default_file_name += ".ply"; // default to ply

    const QString& fileName = QFileDialog::getSaveFileName(
                this,
                "Save file",
                QString::fromStdString(default_file_name),
                "Supported formats (*.ply *.obj *.off *.stl *.sm *.bin *.las *.laz *.xyz *.bxyz *.vg *.bvg *.plm *.pm *.mesh)\n"
                "Surface Mesh (*.ply *.obj *.off *.stl *.sm)\n"
                "Point Cloud (*.ply *.bin *.ptx *.las *.laz *.xyz *.bxyz *.vg *.bvg)\n"
                "Polyhedral Mesh (*.plm *.pm *.mesh)\n"
                "Graph (*.ply)\n"
                "All formats (*.*)"
    );

    if (fileName.isEmpty())
        return false;

    bool saved = false;
    if (dynamic_cast<const PointCloud*>(model)) {
        const PointCloud* cloud = dynamic_cast<const PointCloud*>(model);
        saved = PointCloudIO::save(fileName.toStdString(), cloud);
    }
    else if (dynamic_cast<const SurfaceMesh*>(model)) {
        const SurfaceMesh* mesh = dynamic_cast<const SurfaceMesh*>(model);
        saved = SurfaceMeshIO::save(fileName.toStdString(), mesh);
    }
    else if (dynamic_cast<const Graph*>(model)) {
        const Graph* graph = dynamic_cast<const Graph*>(model);
        saved = GraphIO::save(fileName.toStdString(), graph);
    }
    else if (dynamic_cast<const PolyMesh*>(model)) {
        const PolyMesh* mesh = dynamic_cast<const PolyMesh*>(model);
        saved = PolyMeshIO::save(fileName.toStdString(), mesh);
    }

    if (saved) {
        LOG(INFO) << "model successfully saved to: " << fileName.toStdString();
        setCurrentFile(fileName);
        return true;
    }

    return false;
}


Model* MainWindow::open(const std::string& file_name) {
    auto models = viewer_->models();
    for (auto m : models) {
        if (m->name() == file_name) {
            LOG(WARNING) << "model already loaded: " << file_name;
            return nullptr;
        }
    }

    const std::string& ext = file_system::extension(file_name, true);
    bool is_ply_mesh = false;
    if (ext == "ply")
        is_ply_mesh = (io::PlyReader::num_instances(file_name, "face") > 0);

    Model* model = nullptr;
    if ((ext == "ply" && is_ply_mesh) || ext == "obj" || ext == "off" || ext == "stl" || ext == "sm" || ext == "geojson" || ext == "trilist") { // mesh
        model = SurfaceMeshIO::load(file_name);
    }
    else if (ext == "ply" && io::PlyReader::num_instances(file_name, "edge") > 0) {
        model = GraphIO::load(file_name);
    } else if (ext == "plm" || ext == "pm" || ext == "mesh") {
        model = PolyMeshIO::load(file_name);
    }
    else { // point cloud
        if (ext == "ptx") {
            io::PointCloudIO_ptx serializer(file_name);
            PointCloud* cloud = nullptr;
            while ((cloud = serializer.load_next())) {
                viewer_->addModel(cloud);
                ui->treeWidgetModels->addModel(cloud, true);
            }
        }
        else
            model = PointCloudIO::load(file_name);
    }

    if (model) {
        model->set_name(file_name);
        viewer_->addModel(model);
        ui->treeWidgetModels->addModel(model, true);
        setCurrentFile(QString::fromStdString(file_name));

        const auto keyframe_file = file_system::replace_extension(model->name(), "kf");
        if (file_system::is_file(keyframe_file)) {
            if (viewer_->walkThrough()->interpolator()->read_keyframes(keyframe_file)) {
                LOG(INFO) << "model has an accompanying animation file \'"
                          << file_system::simple_name(keyframe_file) << "\' (loaded)";
                viewer_->walkThrough()->set_scene({model});
            }
        }
    }

    return model;
}


void MainWindow::generateColorPropertyFromIndexedColors() {
    const QString& fileName = QFileDialog::getOpenFileName(
            this,
            "Open file",
            curDataDirectory_,
            "Indexed colors (*.ic)\n"
            "All formats (*.*)"
    );
    if (fileName.isEmpty())
        return;
    // Hide closed dialog
    QApplication::processEvents();

    std::ifstream input(fileName.toStdString().c_str());
    if (input.fail())
        return;

    // I want to handle skip lines and comments starting with '#'
    auto get_line = [](io::LineInputStream &in) -> void {
        in.get_line();
        const char *p = in.current_line().c_str();
        while (!in.eof() && (
                (strlen(p) == 0 || !isprint(*p)) || // empty line
                (strlen(p) > 0 && p[0] == '#')
        )) {
            in.get_line();
            p = in.current_line().c_str();
        }
    };

    std::unordered_map<int, vec3> indexed_colors;
    io::LineInputStream line_input(input);
    int location = -1; // 0: vertex, 1: face
    std::string attribute_name;
    while (!line_input.eof()) {
        get_line(line_input);
        const std::string current_line = line_input.current_line();
        if (current_line.empty())
            continue;
        if (current_line[0] == 'v' || current_line[1] == ':') {
            location = 0;
            line_input >> attribute_name;
        }
        else if (current_line[0] == 'f' || current_line[1] == ':') {
            location = 1;
            line_input >> attribute_name;
        }
        else {
            int index;
            vec3 color;
            line_input >> index >> color;
            if (!line_input.fail())
                indexed_colors[index] = color / 255.0f;
        }
    }

    if (location == 0) {
        if (dynamic_cast<PointCloud*>(viewer_->currentModel())) {
            PointCloud* cloud = dynamic_cast<PointCloud*>(viewer_->currentModel());
            auto indices = cloud->get_vertex_property<int>(attribute_name);
            if (indices) {
                const std::string color_name = "v:color_indexed";
                auto colors = cloud->vertex_property<vec3>(color_name);
                for (auto v : cloud->vertices()) {
                    colors[v] = indexed_colors[indices[v]];
                }
                updateUi();
                LOG(INFO) << "color property '" << color_name << "' has been generated";
            }
        }
        else if (dynamic_cast<SurfaceMesh*>(viewer_->currentModel())) {
            LOG(INFO) << "applying indexed colors on vertices";
            LOG(WARNING) << "not implemented yet... Please remind Liangliang to implement it";
        }
    }
    else if (location == 1) {
        LOG(INFO) << "applying indexed colors on faces";
        LOG(WARNING) << "not implemented yet... Please remind Liangliang to implement it";
    }
    else
        LOG(WARNING) << "unknown location of the indexed colors";
}


void MainWindow::updateUi() {
    const Model* model = viewer_->currentModel();
    if (model) {
        const std::string& name = model->name();
        setCurrentFile(QString::fromStdString(name));
    } else
        updateWindowTitle();

    ui->treeWidgetModels->updateModelList();
    updateRenderingPanel();
    updateStatusBar();
}


void MainWindow::updateRenderingPanel() {
    widgetTrianglesDrawable_->updatePanel();
    widgetLinesDrawable_->updatePanel();
    widgetPointsDrawable_->updatePanel();
}


void MainWindow::activeDrawableChanged(Drawable* d) {
    if (!d)
        return;

    switch (d->type()) {
        case Drawable::DT_POINTS:
            ui->toolBox->setCurrentWidget(ui->toolBoxPointsDrawable);
            dynamic_cast<WidgetDrawable*>(widgetPointsDrawable_)->setActiveDrawable(d);
            break;
        case Drawable::DT_LINES:
            ui->toolBox->setCurrentWidget(ui->toolBoxLinesDrawable);
            dynamic_cast<WidgetDrawable*>(widgetLinesDrawable_)->setActiveDrawable(d);
            break;
        case Drawable::DT_TRIANGLES:
            ui->toolBox->setCurrentWidget(ui->toolBoxTrianglesDrawable);
            dynamic_cast<WidgetDrawable*>(widgetTrianglesDrawable_)->setActiveDrawable(d);
            break;
    }
}


void MainWindow::setCurrentFile(const QString &fileName)
{
    QString dir = fileName.left(fileName.lastIndexOf("/"));
    if (!dir.isEmpty() && file_system::is_directory(dir.toStdString()))
        curDataDirectory_ = dir;

    setWindowModified(false);

    if (!fileName.isEmpty()) {
        recentFiles_.removeAll(fileName);
        recentFiles_.prepend(fileName);
        updateRecentFileActions();
    }

    updateWindowTitle();
}


void MainWindow::enableCameraManipulation() {
    ui->actionCameraManipulation->trigger();
}


void MainWindow::setShowSelectedOnly(bool b) {
    ui->checkBoxSelectedOnly->setChecked(b);
}


WidgetModelList* MainWindow::widgetModelList() const {
    return ui->treeWidgetModels;
}


void MainWindow::onOpenRecentFile() {
    if (okToContinue()) {
        QAction *action = qobject_cast<QAction *>(sender());
        if (action) {
            const QString filename(action->data().toString());
            if (open(filename.toStdString()))
                viewer_->update();
        }
    }
}


void MainWindow::onClearRecentFiles() {
    recentFiles_.clear();
    updateRecentFileActions();
}


void MainWindow::saveSnapshot() {
    DialogSnapshot dialog(this);
    if (dialog.exec() == QDialog::Accepted)
        dialog.saveSnapshot();
}


void MainWindow::setBackgroundColor() {
    const vec4& c = viewer_->backGroundColor();
    QColor orig(static_cast<int>(c.r * 255), static_cast<int>(c.g * 255), static_cast<int>(c.b * 255), static_cast<int>(c.a * 255));
    const QColor& color = QColorDialog::getColor(orig, this);
    if (color.isValid()) {
        const vec4 newColor(color.redF(), color.greenF(), color.blueF(), color.alphaF());
        viewer_->setBackgroundColor(newColor);
        viewer_->update();
    }
}


void MainWindow::saveCameraStateToFile() {
    QString suggested_name = curDataDirectory_;
    if (viewer()->currentModel()) {
        const std::string name = file_system::replace_extension(viewer()->currentModel()->name(), "view");
        suggested_name = QString::fromStdString(name);
    }
    const QString fileName = QFileDialog::getSaveFileName(
            this,
            "Save viewer state to file",
            suggested_name,
            "Viewer state (*.view)\n"
            "All formats (*.*)"
    );

    if (fileName.isEmpty())
        return;

    std::ofstream output(fileName.toStdString().c_str());
    if (output.fail()) {
        QMessageBox::warning(window(), tr("Save state to file error"), tr("Unable to create file %1").arg(fileName));
        return;
    }

    viewer_->saveState(output);
    // assume the user will soon restore the state from this file.
    curDataDirectory_ = fileName.left(fileName.lastIndexOf("/"));
}


void MainWindow::restoreCameraStateFromFile() {
    const QString fileName = QFileDialog::getOpenFileName(
            this,
            "Restore viewer state from file",
            curDataDirectory_,
            "Viewer state (*.view)\n"
            "Viewer state - old deprecated version (*.state)\n"
            "All formats (*.*)"
    );

    if (fileName.isEmpty())
        return;

    // read the state from file
    std::ifstream input(fileName.toStdString().c_str());
    if (input.fail()) {
        QMessageBox::warning(this, tr("Read state file error"), tr("Unable to read file %1").arg(fileName));
        return;
    }

    viewer_->restoreState(input);
}


bool MainWindow::okToContinue()
{
    if (isWindowModified()) {
        int r = QMessageBox::warning(this, tr("Mapple"),
            tr("The model has been modified.\n"
            "Do you want to save your changes?"),
            QMessageBox::Yes | QMessageBox::Default,
            QMessageBox::No,
            QMessageBox::Cancel | QMessageBox::Escape);
        if (r == QMessageBox::Yes)
            return onSave();
        else if (r == QMessageBox::Cancel)
            return false;
    }
    return true;
}


void MainWindow::onAbout()
{
    QString title = QMessageBox::tr(
        "<p align=\"center\"><span style=\"font-style:italic;\">I'm good software, though I have defects.</span></p>"
        );

    int bits = 64;
#if defined (ENV_32_BIT)
    bits = 32;
#endif

#ifndef NDEBUG
    title += QMessageBox::tr("<h3>Mapple (%1 bit) - Debug Version</h3>").arg(bits);
#else
    title += QMessageBox::tr("<h3>Mapple (%1 bit)</h3>").arg(bits);
#endif

    title += QMessageBox::tr("<h4>Version %1</h4>").arg(version().c_str());

    QString text = QMessageBox::tr(
            "<p>Mapple is software for processing and rendering 3D data (e.g., point clouds, graphs, surface meshes, "
            "and polyhedral meshes), and more.</p>"
            "<p>Liangliang Nan<br>"
            "<a href=\"mailto:liangliang.nan@gmail.com\">liangliang.nan@gmail.com</a><br>"
            "<a href=\"https://3d.bk.tudelft.nl/liangliang/\">https://3d.bk.tudelft.nl/liangliang/</a></p>"
    );

    //QMessageBox::about(this, title, text);
    QMessageBox::about(this, "About Mapple", title + text);
}


void MainWindow::Save_Plane_Param() {
    Yaml::Node params_;
    Yaml::Parse (params_, "./params.yaml");

    float new_plane_d = params_["new_plane_d"].As<float> ();
    bool new_plane = params_["new_plane"].As<bool> ();
    bool two_plane = params_["two_plane"].As<bool> ();

    auto my_cloud = dynamic_cast<PointCloud*>(viewer_->currentModel());
    if (my_cloud)
    {

        ::internal::EnrichedPointCloud point_set(my_cloud, my_cloud->vertex_property<int>("v:primitive_index"));
        // const std::vector<::internal::Planar_segment *> &planar_segments = point_set.planar_segments();

        auto hypothesis_ = new ::internal::Hypothesis;
        hypothesis_->generate(point_set);

        auto &planes = hypothesis_->supporting_planes();
        LOG(INFO) << planes.size() << " planes generated";

        // when we need a new Plane3
        vec3 center_new, center_temp;
        float center_dist;
        vec3 new_normal;

        auto &centers = hypothesis_->supporting_planes_center(); 

        auto &this_plane =  planes[0];
        auto &this_plane_center =  centers[0];
        auto dist_max = hypothesis_->supporting_segments()[0]->plane_deviation();
        LOG(INFO) << "plane_deviation: " << dist_max;

        if (new_plane) {
            float orient = this_plane->orient(vec3(0, 0, 0)) ;

            center_temp = centers[0] + new_plane_d * this_plane->normal();
            float orient_temp = this_plane->orient(center_temp) ;
            if (orient_temp * orient > 0)
                center_new = center_temp;
            else
                center_new = centers[0] - new_plane_d * this_plane->normal();
        } else if (two_plane) {
            // center_temp = this_plane->projection(centers[1]);
            // center_new = (center_temp + centers[1]) * 0.5f;

            // float thetra = geom::to_degrees(geom::angle2(planes[0]->normal(), planes[1]->normal()));
            // LOG(INFO) << "two_plane_normal_deviation: " << thetra;

            // center_dist = this_plane->squared_distance(centers[1]);
            // LOG(INFO) << "two_plane_center_distance: " << sqrt(center_dist);
            ////////////////////////////////////////////////////////////////////////////////////////////
            Line3 line_intersect;
            
            bool ok;
            ok = this_plane->intersect(*planes[1], line_intersect);
            if (!ok)
                LOG(ERROR) << "two_plane not intersect";
            float thetra = geom::to_degrees(geom::angle(planes[0]->normal(), planes[1]->normal()));

            if (thetra > 90)
                new_normal = -planes[0]->normal() + planes[1]->normal();
            else
                new_normal = planes[0]->normal() + planes[1]->normal();
            new_normal.normalize();
            center_new = line_intersect.point() + new_plane_d * new_normal;
            LOG(INFO) <<  new_plane_d << " " << new_normal;
        }

        // auto &meshes = hypothesis_->supporting_meshes();
        // if (meshes.size() != planes.size()){
        //     LOG(ERROR) << "meshes.size() != planes.size()";
        //     return;
        // }
        // string::replace(cam.image_file, "_cam.txt", ".jpg");
        // std::string file_name = getCurDataDirectory().toStdString() + "/calib.txt";


        std::string file_name = my_cloud->name();
        string::replace(file_name, ".ply", ".txt");
        std::ofstream output(file_name.c_str());
        if (output.fail()) {
            LOG(ERROR) << "could not open file: " << file_name;
            return ;
        }
        output.precision(10);

        for (std::size_t i = 0; i < planes.size(); i++) {
            // SurfaceMeshPolygonization polygonizer;
            // polygonizer.apply(meshes[i]);
            auto mesh = hypothesis_->supporting_segments()[i]->borders();
            viewer_->addModel(mesh);
        }   

        if (new_plane)
            output << center_new << " " << this_plane->normal() << std::endl;
        else if (two_plane)
            output << center_new << " " << new_normal << std::endl;
        else            
            output << this_plane_center << " " << this_plane->normal() << std::endl;

        if (new_plane) {
            Plane3 *new_plane = new Plane3(center_new, this_plane->normal());
            auto new_mesh = hypothesis_->supporting_segments()[0]->borders(new_plane);
            viewer_->addModel(new_mesh);
        }
        else if (two_plane) {
            Plane3 *new_plane = new Plane3(center_new, new_normal);
            auto new_mesh = hypothesis_->supporting_segments()[0]->borders(new_plane);
            viewer_->addModel(new_mesh);
        }

        // LOG(INFO) << getCurDataDirectory().toStdString() ;
        LOG(INFO) << file_name;
        

        updateUi();
        viewer_->update();
    }
}
void MainWindow::showManual() {
    // auto cloud = new PointCloud;
    // auto colors = cloud->add_vertex_property<vec3>("v:color");
    // for (int i = 0; i < 100; ++i) {
    //     auto v = cloud->add_vertex(easy3d::vec3(random_float(), random_float(), random_float()));
    //     colors[v] = vec3(1.0f, 0.0f, 0.0f);
    // }
    // viewer()->addModel(cloud);
    // auto drawable = cloud->renderer()->get_points_drawable("vertices");
    // drawable->set_coloring(Drawable::COLOR_PROPERTY, Drawable::VERTEX, "v:color");
    // drawable->set_point_size(10.0f);

    // Timer<PaintCanvas *> timer;
    // timer.set_interval(300, edit_model, viewer());
    // Timer<>::single_shot(200000, &timer, &Timer<PaintCanvas *>::stop);

    // return ;

    // const std::vector<vec3> &points = resource::bunny_vertices;

    // auto cloud = new PointCloud;
    // cloud->resize(points.size());
    // auto points_cloud = cloud->vertex_property<vec3>("v:point");
    // auto &positions = points_cloud.vector();
    // for (auto &p : positions)
    //     p = vec3(0);
    // cloud->set_name("zhl");
    // viewer()->addModel(cloud);

    // auto drawable = cloud->renderer()->get_points_drawable("vertices");
    // viewer()->makeCurrent();
    // drawable->update_vertex_buffer(points, true);
    // viewer()->doneCurrent();


    // updateUi();
    // viewer()->update();

    // return ;

    Graph* outcome = new Graph;
    outcome->set_name("outcome");
    Graph* center_v = new Graph; 
    center_v->set_name("center_v");
    Graph* Plane_plus = new Graph; 
    Plane_plus->set_name("Plane_plus");

    outcome->add_vertex(vec3(0));
    center_v->add_vertex(vec3(0));
    Plane_plus->add_vertex(vec3(0));

    // std::vector<Graph::Vertex> g_edges, gg_edges;
    // std::vector<Graph::Vertex> center_v_edges;
    // std::vector<Graph::Vertex> Plane_plus_edges;

    // center_v_edges.push_back(center_v->add_vertex(vec3(0)));
    // center_v_edges.push_back(center_v->add_vertex(vec3(0)));
    // center_v->add_edge(center_v_edges[0], center_v_edges[1]);

    // Plane_plus_edges.push_back(Plane_plus->add_vertex(vec3(0)));
    // Plane_plus_edges.push_back(Plane_plus->add_vertex(vec3(0)));
    // Plane_plus->add_edge(Plane_plus_edges[0], Plane_plus_edges[1]);

    // for (std::size_t i = 0; i < 6; i++) {
    //     g_edges.push_back(outcome->add_vertex(vec3(0)));
    //     gg_edges.push_back(outcome->add_vertex(vec3(0)));
    // }

    // outcome->add_edge(g_edges[0], g_edges[1]);
    // outcome->add_edge(g_edges[1], g_edges[2]);
    // outcome->add_edge(g_edges[2], g_edges[3]);
    // outcome->add_edge(g_edges[3], g_edges[0]);
    // outcome->add_edge(g_edges[4], g_edges[5]);

    // outcome->add_edge(gg_edges[0], gg_edges[1]);
    // outcome->add_edge(gg_edges[1], gg_edges[2]);
    // outcome->add_edge(gg_edges[2], gg_edges[3]);
    // outcome->add_edge(gg_edges[3], gg_edges[0]);
    // outcome->add_edge(gg_edges[4], gg_edges[5]);

    viewer_->addModel(outcome);
    viewer_->addModel(center_v);
    viewer_->addModel(Plane_plus);

    auto edges = outcome->renderer()->get_lines_drawable("edges");
    edges->set_uniform_coloring(vec4(0, 1, 0, 1));
    auto vertices = outcome->renderer()->get_points_drawable("vertices");
    vertices->set_uniform_coloring(vec4(1, 0, 0, 1));
    vertices->set_point_size(10.0f);
    auto Plane_plus_vertices = Plane_plus->renderer()->get_points_drawable("vertices");
    Plane_plus_vertices->set_point_size(9.0f);
    Plane_plus->renderer()->set_visible(false);


    updateUi();
    viewer_->update();

    return ;


    // const float theta = static_cast<float>(3.1415926 * 20.0 / 180.0f);  

    auto my_cloud = dynamic_cast<PointCloud*>(viewer_->currentModel());
    if (my_cloud)
    {

        ::internal::EnrichedPointCloud point_set(my_cloud, my_cloud->vertex_property<int>("v:primitive_index"));
        // const std::vector<::internal::Planar_segment *> &planar_segments = point_set.planar_segments();

        auto hypothesis_ = new ::internal::Hypothesis;
        hypothesis_->generate(point_set);

        auto &planes = hypothesis_->supporting_planes();
        LOG(INFO) << planes.size() << " planes generated";

        for (std::size_t i = 1; i < planes.size(); i++) {
            LOG(INFO) << geom::to_degrees( geom::angle2(planes[i]->normal(), planes[0]->normal()) ) ;
        }

        Line3 l42, line_h, line_v;
        Line3 line_bottom, line_left, line_right, line_top;
    
        // bool ok;
        // ok = planes[0]->intersect(*planes[1], l42);
        // LOG(INFO) << "l42 ok: " << ok;
        // calculate height
        // auto v_height_z = my_cloud->vertex_property<float>("v:height_plane");
        // for (auto v : my_cloud->vertices()) {
        //     const auto& p = my_cloud->position(v);
        //     float d = planes[0]->squared_distance(p);
        //     if (d > 0.49) d = 0.49;
        //     if (planes[0]->orient(p) < 0)
        //         d *= -1;
        //         v_height_z[v] = d;
        // }
        // LOG(INFO) << "calculate distance to plane done. ";

        // auto &meshes = hypothesis_->supporting_meshes();
        // if (meshes.size() != planes.size()){
        //     LOG(INFO) << "meshes.size() != planes.size()";
            // return;
        // }

        // auto &centers = hypothesis_->supporting_planes_center();
        // float d0_squared = l42.squared_distance(centers[0]);

        // Plane3 *plane_yoz;
        // plane_yoz = new Plane3(vec3(0, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1));
        // for (std::size_t i = 1; i < meshes.size(); i++) {
        //     LOG(INFO) << "plane_yoz_orient: " << plane_yoz->orient(centers[i]);
        // }

        // std::vector<vec3> interpoint;
        // float height_squared = 0;

        // for (std::size_t i = 2; i < meshes.size(); i++) {
        //     float height_temp = planes[0]->squared_distance(centers[i]);
        //     if (height_squared < height_temp)
        //         height_squared = height_temp;
        // }
        // LOG(INFO) << "LanGao: " << 2*sqrt(height_squared);

        // for (std::size_t i = 0; i < meshes.size(); i++) {
            // SurfaceMeshPolygonization polygonizer;
            // polygonizer.apply(meshes[i]);

            // float dd = planes[i]->squared_distance(vec3(0, 0, 0));
            // LOG(INFO) << "LanGao: " << sqrt(dd) ;

            // std::string file_name = getCurDataDirectory().toStdString() + "/calib.txt";
            // std::ofstream output(file_name.c_str());
            // if (output.fail()) {
            //     LOG(ERROR) << "could not open file: " << file_name;
            //     return ;
            // }
            // output.precision(16);
            // output << planes[i]->point() << "   " << planes[i]->normal();
            // LOG(INFO) << getCurDataDirectory().toStdString() ;

        //     viewer_->addModel(meshes[i]);
        // }

        // for (std::size_t i = 0; i < 1; i++) {
        //     SurfaceMeshPolygonization polygonizer;
        //     polygonizer.apply(meshes[i]);
        //     viewer_->addModel(meshes[i]);

        //     std::vector<vec3> interpoint_v, interpoint_h;
            
        //     Plane3* plane_v;
        //     Plane3* plane_h;
        //     Plane3* plane_bottom;
        //     if (i == 0) {
        //         plane_v = new Plane3(centers[i], planes[1]->normal());
        //         plane_h = new Plane3(centers[i], cross(planes[0]->normal(), planes[1]->normal()));
        //     } else {
        //         plane_v = new Plane3(centers[i], planes[0]->normal());
        //         plane_h = new Plane3(centers[i], cross(planes[0]->normal(), planes[1]->normal()));
        //     }

        //     bool orient_h;
        //     if (dot(plane_h->normal(), vec3(0, 1, 0)) > 0)
        //         orient_h = true;
        //     else
        //         orient_h = false;
        //     // LOG(INFO) << " orient_h: " << orient_h << true;
        //     // dir = cross(l42.direction(), planes[i]->normal());

        //     std::vector<double> dist_h1, dist_h2;
        //     std::vector<vec3> points_v, points_h1, points_h2;
        //     std::vector<double> dist_v;

        //     ok = planes[0]->intersect(*plane_h, line_h);

        //     for (auto e : meshes[i]->edges()){
        //         const SurfaceMesh::Halfedge h0 = meshes[i]->halfedge(e, 0);
        //         const SurfaceMesh::Halfedge h1 = meshes[i]->halfedge(e, 1);

        //         const vec3 p0 = (vec3) meshes[i]->position(meshes[i]->target(h0));
        //         const vec3 p1 = (vec3) meshes[i]->position(meshes[i]->target(h1));

        //         bool parallel_h1 = std::abs(dot(Line3::from_two_points(p0, p1).direction(), line_h.direction())) > std::cos(theta);
        //         if (plane_h->orient(p0) >0 && parallel_h1){
        //             float dist0 = line_h.squared_distance(p0);
        //             float dist1 = line_h.squared_distance(p1);
        //             LOG(INFO) << " parallel_h1: " << dist0 << "  " << dist1;
        //             dist_h1.push_back(dist0);
        //             dist_h1.push_back(dist1);
        //             points_h1.push_back(p0);
        //             points_h1.push_back(p1);
        //         }

        //         bool parallel_h2 = std::abs(dot(Line3::from_two_points(p0, p1).direction(), line_h.direction())) > std::cos(theta);
        //         if (plane_h->orient(p0) <0 && parallel_h2){
        //             float dist0 = line_h.squared_distance(p0);
        //             float dist1 = line_h.squared_distance(p1);
        //             LOG(INFO) << " parallel_h2: " << dist0 << "  " << dist1;
        //             dist_h2.push_back(dist0);
        //             dist_h2.push_back(dist1);
        //             points_h2.push_back(p0);
        //             points_h2.push_back(p1);
        //         }

        //         bool parallel_v = std::abs(dot(Line3::from_two_points(p0, p1).direction(), l42.direction())) > std::cos(theta);
        //         if (parallel_v){
        //             float dist0 = l42.squared_distance(p0);
        //             float dist1 = l42.squared_distance(p1);
        //             if (dist0 > d0_squared) {
        //                 LOG(INFO) << " parallel_v: " << dist0 << "  " << dist1;
        //                 dist_v.push_back(dist0);
        //                 dist_v.push_back(dist1);
        //                 points_v.push_back(p0);
        //                 points_v.push_back(p1);
        //             }
        //         }

        //         // LOG(INFO) << " p0: " << p0 << " p1: " << p1;

        //         // if (plane_v->intersect(p0, p1, temp)){
        //         //     if (std::find(interpoint_v.begin(), interpoint_v.end(), temp) != interpoint_v.end()) ;
        //         //     else    {
        //         //         interpoint_v.push_back(temp);
        //         //         interpoint.push_back(temp);
        //         //     }
        //         //     LOG(INFO) << "intersect_v: " << temp;
        //         // }
        //         // if (plane_h->intersect(p0, p1, temp)){
        //         //     if (std::find(interpoint_h.begin(), interpoint_h.end(), temp) != interpoint_h.end()) ;
        //         //     else    {
        //         //         interpoint_h.push_back(temp);
        //         //         interpoint.push_back(temp);
        //         //     }
        //         //     LOG(INFO) << "intersect_h: " << temp;
        //         // }
        //     }  // end of edges

        //     std::vector<double>::iterator min_h1 = std::min_element(dist_h1.begin(), dist_h1.end());
        //     line_left.set(points_h1[std::distance(dist_h1.begin(), min_h1)], line_h.direction());
        //     LOG(INFO) << "min_h1: " << sqrt(*min_h1);
        //     std::vector<double>::iterator min_h2 = std::min_element(dist_h2.begin(), dist_h2.end());
        //     line_right.set(points_h2[std::distance(dist_h2.begin(), min_h2)], line_h.direction());
        //     LOG(INFO) << "min_h2: " << sqrt(*min_h2);
        //     LOG(INFO) << "min_h: " << sqrt(*min_h2) + sqrt(*min_h1);
        //     std::vector<double>::iterator min_v = std::min_element(dist_v.begin(), dist_v.end());
        //     LOG(INFO) << "min_v: " << sqrt(*min_v);
        //     line_bottom.set(points_v[std::distance(dist_v.begin(), min_v)], l42.direction());
        //     plane_bottom = new Plane3(points_v[std::distance(dist_v.begin(), min_v)], planes[1]->normal());

        //     // if (interpoint_v.size() == 2){
        //     //     LOG(INFO) << "distance v= " << distance(interpoint_v[0], interpoint_v[1]) ;
        //     // }
        //     // if (interpoint_h.size() == 2){
        //     //     float d0, d1, d;
        //     //     d0 = (float) l42.squared_distance(interpoint_h[0]);
        //     //     d1 = (float) l42.squared_distance(interpoint_h[1]);
        //     //     d  = (d0 > d1) ? sqrt(d0) : sqrt(d1);
        //     //     LOG(INFO) << "distance h= " << d;
        //     // }
        //     vec3 temp;

        //     if (planes[1]->intersect(line_left, temp))
        //         interpoint.push_back(temp); 
        //     if (planes[1]->intersect(line_right, temp))
        //         interpoint.push_back(temp);
        //     if (plane_bottom->intersect(line_right, temp))
        //         interpoint.push_back(temp);
        //     if (plane_bottom->intersect(line_left, temp))
        //         interpoint.push_back(temp);

        // } // end of each mesh

        // Plane3 *plane_center_v;
        // vec3 point_l42, point_bottom;
        // double angle_center_v;
        // plane_center_v = new Plane3(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 0, 1));
        // if (plane_center_v->intersect(l42, point_l42) && plane_center_v->intersect(line_bottom, point_bottom))
        //     LOG(INFO) << "plane_center_v: sucess";
        // angle_center_v = acos(std::abs(dot(normalize(point_l42 - point_bottom), line_left.direction())));
        // angle_center_v = angle_center_v * 180.0 /3.1415926 ;
        // LOG(INFO) << "angle_center_v: " << angle_center_v;

        // float del1 = sqrt(line_left.squared_distance(point_l42));
        // float del2 = sqrt(line_left.squared_distance(point_bottom));
        // LOG(INFO) << del1 << "  " << del2;

        // std::vector<Graph::Vertex> g_edges;
        // Graph* outcome = new Graph; 
        // outcome->set_name("outcome");
        // for (std::size_t i = 0; i < interpoint.size(); i++) {
        //     g_edges.push_back(outcome->add_vertex(interpoint[i]));
        // }
        // g_edges.push_back(outcome->add_vertex(point_l42));
        // g_edges.push_back(outcome->add_vertex(point_bottom));

        // if (interpoint.size() == 4) {
        //     outcome->add_edge(g_edges[0], g_edges[1]);
        //     outcome->add_edge(g_edges[1], g_edges[2]);
        //     outcome->add_edge(g_edges[2], g_edges[3]);
        //     outcome->add_edge(g_edges[3], g_edges[0]);
        // }
        // outcome->add_edge(g_edges[4], g_edges[5]);

        // viewer_->addModel(outcome);

        updateUi();
        viewer_->update();

    }

}

void MainWindow::show_boundary() {

    const float theta = static_cast<float>(3.1415926 * 3.0 / 180.0f);  

    auto my_cloud = dynamic_cast<PointCloud*>(viewer_->currentModel());
    if (my_cloud)
    {

        ::internal::EnrichedPointCloud point_set(my_cloud, my_cloud->vertex_property<int>("v:primitive_index"));
        // const std::vector<::internal::Planar_segment *> &planar_segments = point_set.planar_segments();

        auto hypothesis_ = new ::internal::Hypothesis;
        hypothesis_->generate(point_set);
        auto &planes = hypothesis_->supporting_planes();
        LOG(INFO) << planes.size() << " planes generated";

        Line3 l42, line_h, line_v;
        Line3 line_bottom, line_left, line_right, line_top;
   
        bool ok;
        ok = planes[0]->intersect(*planes[1], l42);
        LOG(INFO) << "l42 ok: " << ok;

        // calculate height
        // auto v_height_z = my_cloud->vertex_property<float>("v:height_plane");
        // for (auto v : my_cloud->vertices()) {
        //     const auto& p = my_cloud->position(v);
        //     float d = planes[0]->squared_distance(p);
        //     if (d > 0.49) d = 0.49;
        //     if (planes[0]->orient(p) < 0)
        //         d *= -1;
        //         v_height_z[v] = d;
        // }
        // LOG(INFO) << "calculate distance to plane done. ";

        auto &meshes = hypothesis_->supporting_meshes();
        if (meshes.size() != planes.size()){
            LOG(INFO) << "meshes.size() != planes.size()";
            return;
        }

        auto &centers = hypothesis_->supporting_planes_center();
        float d0_squared = l42.squared_distance(centers[0]);

        Plane3 *plane_yoz;
        plane_yoz = new Plane3(vec3(0, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1));
        for (std::size_t i = 1; i < meshes.size(); i++) {
            LOG(INFO) << "plane_yoz_orient: " << plane_yoz->orient(centers[i]);
        }

        std::vector<vec3> interpoint;
        float height_squared = 0;

        for (std::size_t i = 2; i < meshes.size(); i++) {
            float height_temp = planes[0]->squared_distance(centers[i]);
            if (height_squared < height_temp)
                height_squared = height_temp;
        }
        LOG(INFO) << "LanGao: " << 2*sqrt(height_squared);

        for (std::size_t i = 1; i < meshes.size(); i++) {
            SurfaceMeshPolygonization polygonizer;
            polygonizer.apply(meshes[i]);
            viewer_->addModel(meshes[i]);
        }

        for (std::size_t i = 0; i < 1; i++) {
            SurfaceMeshPolygonization polygonizer;
            polygonizer.apply(meshes[i]);
            viewer_->addModel(meshes[i]);

            std::vector<vec3> interpoint_v, interpoint_h;
            
            Plane3* plane_v;
            Plane3* plane_h;
            Plane3* plane_bottom;
            Plane3* plane_top;
            if (i == 0) {
                plane_v = new Plane3(centers[i], planes[1]->normal());
                plane_h = new Plane3(centers[i], cross(planes[0]->normal(), planes[1]->normal()));
            } else {
                plane_v = new Plane3(centers[i], planes[0]->normal());
                plane_h = new Plane3(centers[i], cross(planes[0]->normal(), planes[1]->normal()));
            }

            bool orient_h, orient_v;
            if (dot(plane_h->normal(), vec3(0, 1, 0)) > 0)
                orient_h = true;
            else
                orient_h = false;
            if (dot(plane_v->normal(), vec3(1, 0, 0)) > 0)
                orient_v = true;
            else
                orient_v = false;
            // LOG(INFO) << " orient_h: " << orient_h << true;
            // dir = cross(l42.direction(), planes[i]->normal());

            std::vector<double> dist_h1, dist_h2;
            std::vector<vec3> points_v1, points_v2, points_h1, points_h2;
            std::vector<double> dist_v1, dist_v2;

            ok = planes[0]->intersect(*plane_h, line_h);
            ok = planes[0]->intersect(*plane_v, line_v);

            for (auto e : meshes[i]->edges()){
                const SurfaceMesh::Halfedge h0 = meshes[i]->halfedge(e, 0);
                const SurfaceMesh::Halfedge h1 = meshes[i]->halfedge(e, 1);

                const vec3 p0 = (vec3) meshes[i]->position(meshes[i]->target(h0));
                const vec3 p1 = (vec3) meshes[i]->position(meshes[i]->target(h1));

                bool parallel_h1 = std::abs(dot(Line3::from_two_points(p0, p1).direction(), line_h.direction())) > std::cos(theta);
                if (plane_h->orient(p0) >0 && parallel_h1){
                    float dist0 = line_h.squared_distance(p0);
                    float dist1 = line_h.squared_distance(p1);
                    LOG(INFO) << " parallel_h1: " << dist0 << "  " << dist1;
                    dist_h1.push_back(dist0);
                    dist_h1.push_back(dist1);
                    points_h1.push_back(p0);
                    points_h1.push_back(p1);
                }

                bool parallel_h2 = std::abs(dot(Line3::from_two_points(p0, p1).direction(), line_h.direction())) > std::cos(theta);
                if (plane_h->orient(p0) <0 && parallel_h2){
                    float dist0 = line_h.squared_distance(p0);
                    float dist1 = line_h.squared_distance(p1);
                    LOG(INFO) << " parallel_h2: " << dist0 << "  " << dist1;
                    dist_h2.push_back(dist0);
                    dist_h2.push_back(dist1);
                    points_h2.push_back(p0);
                    points_h2.push_back(p1);
                }

                bool parallel_v = std::abs(dot(Line3::from_two_points(p0, p1).direction(), l42.direction())) > std::cos(theta);
                if (plane_v->orient(p0) >0 && parallel_v){
                    float dist0 = line_v.squared_distance(p0);
                    float dist1 = line_v.squared_distance(p1);
                    LOG(INFO) << " parallel_v1: " << dist0 << "  " << dist1;
                    dist_v1.push_back(dist0);
                    dist_v1.push_back(dist1);
                    points_v1.push_back(p0);
                    points_v1.push_back(p1);
                } else if (plane_v->orient(p0) <0 && parallel_v) {
                    float dist0 = line_v.squared_distance(p0);
                    float dist1 = line_v.squared_distance(p1);
                    LOG(INFO) << " parallel_v2: " << dist0 << "  " << dist1;
                    dist_v2.push_back(dist0);
                    dist_v2.push_back(dist1);
                    points_v2.push_back(p0);
                    points_v2.push_back(p1);
                }

                // LOG(INFO) << " p0: " << p0 << " p1: " << p1;

                // if (plane_v->intersect(p0, p1, temp)){
                //     if (std::find(interpoint_v.begin(), interpoint_v.end(), temp) != interpoint_v.end()) ;
                //     else    {
                //         interpoint_v.push_back(temp);
                //         interpoint.push_back(temp);
                //     }
                //     LOG(INFO) << "intersect_v: " << temp;
                // }
                // if (plane_h->intersect(p0, p1, temp)){
                //     if (std::find(interpoint_h.begin(), interpoint_h.end(), temp) != interpoint_h.end()) ;
                //     else    {
                //         interpoint_h.push_back(temp);
                //         interpoint.push_back(temp);
                //     }
                //     LOG(INFO) << "intersect_h: " << temp;
                // }
            }  // end of edges

            std::vector<double>::iterator min_h1 = std::min_element(dist_h1.begin(), dist_h1.end());
            LOG(INFO) << "min_h1: " << sqrt(*min_h1);
            std::vector<double>::iterator min_h2 = std::min_element(dist_h2.begin(), dist_h2.end());
            LOG(INFO) << "min_h2: " << sqrt(*min_h2);
            LOG(INFO) << "min_h: " << sqrt(*min_h2) + sqrt(*min_h1);

            if (orient_h == true) {
                line_left.set(points_h1[std::distance(dist_h1.begin(), min_h1)], line_h.direction());
                line_right.set(points_h2[std::distance(dist_h2.begin(), min_h2)], line_h.direction());
            } else {
                line_right.set(points_h1[std::distance(dist_h1.begin(), min_h1)], line_h.direction());
                line_left.set(points_h2[std::distance(dist_h2.begin(), min_h2)], line_h.direction());
            }


            std::vector<double>::iterator min_v1 = std::min_element(dist_v1.begin(), dist_v1.end());
            LOG(INFO) << "min_v1: " << sqrt(*min_v1);
            std::vector<double>::iterator min_v2 = std::min_element(dist_v2.begin(), dist_v2.end());
            LOG(INFO) << "min_v2: " << sqrt(*min_v2);
            LOG(INFO) << "min_v: " << sqrt(*min_v2) + sqrt(*min_v1);

            if (orient_v == true) {
                line_bottom.set(points_v1[std::distance(dist_v1.begin(), min_v1)], l42.direction());
                plane_bottom = new Plane3(points_v1[std::distance(dist_v1.begin(), min_v1)], planes[1]->normal());
                line_top.set(points_v2[std::distance(dist_v2.begin(), min_v2)], l42.direction());
                plane_top = new Plane3(points_v2[std::distance(dist_v2.begin(), min_v2)], planes[1]->normal());
            } else {
                line_top.set(points_v1[std::distance(dist_v1.begin(), min_v1)], l42.direction());
                plane_top = new Plane3(points_v1[std::distance(dist_v1.begin(), min_v1)], planes[1]->normal());
                line_bottom.set(points_v2[std::distance(dist_v2.begin(), min_v2)], l42.direction());
                plane_bottom = new Plane3(points_v2[std::distance(dist_v2.begin(), min_v2)], planes[1]->normal());
            }

            // if (interpoint_v.size() == 2){
            //     LOG(INFO) << "distance v= " << distance(interpoint_v[0], interpoint_v[1]) ;
            // }
            // if (interpoint_h.size() == 2){
            //     float d0, d1, d;
            //     d0 = (float) l42.squared_distance(interpoint_h[0]);
            //     d1 = (float) l42.squared_distance(interpoint_h[1]);
            //     d  = (d0 > d1) ? sqrt(d0) : sqrt(d1);
            //     LOG(INFO) << "distance h= " << d;
            // }
            vec3 temp;

            if (plane_top->intersect(line_left, temp))
                interpoint.push_back(temp); 
            if (plane_top->intersect(line_right, temp))
                interpoint.push_back(temp);
            if (plane_bottom->intersect(line_right, temp))
                interpoint.push_back(temp);
            if (plane_bottom->intersect(line_left, temp))
                interpoint.push_back(temp);

        } // end of each mesh

        Plane3 *plane_center_v;
        vec3 point_top, point_bottom;
        vec3 point_mid_bottom;
        double angle_center_v;
        plane_center_v = new Plane3(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 0, 1));
        if (plane_center_v->intersect(line_top, point_top) && plane_center_v->intersect(line_bottom, point_bottom))
            LOG(INFO) << "plane_center_v: sucess";
        angle_center_v = acos(std::abs(dot(normalize(point_top - point_bottom), line_left.direction())));
        angle_center_v = angle_center_v * 180.0 /3.1415926 ;
        LOG(INFO) << "angle_center_v: " << angle_center_v;
        // LOG(INFO) << "distance: " << easy3d::distance(interpoint[0], interpoint[1]);
        float del1 = sqrt(line_left.squared_distance(point_top));
        float del2 = sqrt(line_left.squared_distance(point_bottom));
        // LOG(INFO) << del1 << "  " << del2;
        if (del1 > del2)
            angle_center_v *= -1;
        point_mid_bottom = (interpoint[2] + interpoint[3]) / 2.0;

        // calculate last
        std::stringstream stream;
        stream << "rect info: "     << std::endl;
        stream << std::fixed << std::setprecision(3) ;
        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "left_top: " << "(" << interpoint[0] << ")"<< std::endl;
        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "right_top: " << "(" << interpoint[1] << ")"<< std::endl;
        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "right_bottom: " << "(" << interpoint[2] << ")"<< std::endl;
        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "left_bottom: " << "(" << interpoint[3] << ")"<< std::endl;

        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "length: " << easy3d::distance(interpoint[0], interpoint[3]) << "m"<< std::endl;
        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "width: "  << easy3d::distance(interpoint[0], interpoint[1]) << "m"<< std::endl;
        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "barrier height: "  << 2*sqrt(height_squared) << "m"<< std::endl;
        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "distane of lidar to plane: " << sqrt(planes[0]->squared_distance(vec3(0,0,0))) << "m"<< std::endl;
        stream << std::fixed << std::setprecision(2) ;
        stream << std::setw(14)<< " " << "deflection angle from center_line: " << angle_center_v << "degree"<< std::endl;
        LOG(WARNING) << stream.str();

        std::vector<Graph::Vertex> g_edges;
        std::vector<Graph::Vertex> center_v_edges;
        Graph* outcome = new Graph; 
        Graph* center_v = new Graph; 
        outcome->set_name("outcome");
        center_v->set_name("center_v");
        for (std::size_t i = 0; i < interpoint.size(); i++) {
            g_edges.push_back(outcome->add_vertex(interpoint[i]));
        }
        g_edges.push_back(outcome->add_vertex(point_top));
        g_edges.push_back(outcome->add_vertex(point_bottom));
        g_edges.push_back(outcome->add_vertex(point_mid_bottom));

        center_v_edges.push_back(center_v->add_vertex(point_top));
        center_v_edges.push_back(center_v->add_vertex(point_bottom));

        if (interpoint.size() == 4) {
            outcome->add_edge(g_edges[0], g_edges[1]);
            outcome->add_edge(g_edges[1], g_edges[2]);
            outcome->add_edge(g_edges[2], g_edges[3]);
            outcome->add_edge(g_edges[3], g_edges[0]);
        }
        // outcome->add_edge(g_edges[4], g_edges[5]);
        outcome->add_edge(g_edges[4], g_edges[6]);

        center_v->add_edge(center_v_edges[0], center_v_edges[1]);

        viewer_->addModel(outcome);
        viewer_->addModel(center_v);

        updateUi();
        viewer_->update();

    }
}

void MainWindow::show_bbox() {

    const float theta = static_cast<float>(3.1415926 * 3.0 / 180.0f);  

    auto my_cloud = dynamic_cast<PointCloud*>(viewer_->currentModel());
    if (my_cloud)
    {

        ::internal::EnrichedPointCloud point_set(my_cloud, my_cloud->vertex_property<int>("v:primitive_index"));

        auto hypothesis_ = new ::internal::Hypothesis;
        hypothesis_->generate(point_set);

        auto &planes = hypothesis_->supporting_planes();
        LOG(INFO) << planes.size() << " planes generated";

        Line3 l42, line_h, line_v;
        Line3 line_bottom, line_left, line_right, line_top;
   
        bool ok;
        ok = planes[0]->intersect(*planes[1], l42);
        LOG(INFO) << "l42 ok: " << ok;

        // calculate height
        // auto v_height_z = my_cloud->vertex_property<float>("v:height_plane");
        // for (auto v : my_cloud->vertices()) {
        //     const auto& p = my_cloud->position(v);
        //     float d = planes[0]->squared_distance(p);
        //     if (d > 0.49) d = 0.49;
        //     if (planes[0]->orient(p) < 0)
        //         d *= -1;
        //         v_height_z[v] = d;
        // }
        // LOG(INFO) << "calculate distance to plane done. ";

        auto &meshes = hypothesis_->supporting_meshes();
        if (meshes.size() != planes.size()){
            LOG(INFO) << "meshes.size() != planes.size()";
            return;
        }

        // for (std::size_t i = 1; i < planes.size(); i++) {

        // }

        auto &centers = hypothesis_->supporting_planes_center();

        float d0_squared = planes[1]->squared_distance(centers[4]);
        LOG(INFO) << "bbox width: " << sqrt(d0_squared);

        // Plane3 *plane_yoz;
        // plane_yoz = new Plane3(vec3(0, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1));
        // for (std::size_t i = 1; i < meshes.size(); i++) {
        //     LOG(INFO) << "plane_yoz_orient: " << plane_yoz->orient(centers[i]);
        // }

        std::vector<vec3> interpoint;
        float height_squared = 0;

        // for (std::size_t i = 2; i < meshes.size(); i++) {
        //     float height_temp = planes[0]->squared_distance(centers[i]);
        //     if (height_squared < height_temp)
        //         height_squared = height_temp;
        // }
        // LOG(INFO) << "LanGao: " << 2*sqrt(height_squared);

        for (std::size_t i = 1; i < meshes.size(); i++) {
            SurfaceMeshPolygonization polygonizer;
            polygonizer.apply(meshes[i]);
            viewer_->addModel(meshes[i]);
        }

        for (std::size_t i = 0; i < 1; i++) {
            SurfaceMeshPolygonization polygonizer;
            polygonizer.apply(meshes[i]);
            viewer_->addModel(meshes[i]);

            std::vector<vec3> interpoint_v, interpoint_h;
            
            Plane3* plane_v;
            Plane3* plane_h;
            Plane3* plane_bottom;
            Plane3* plane_top;
            if (i == 0) {
                plane_h = new Plane3(centers[i], planes[1]->normal());
                plane_v = new Plane3(centers[i], cross(planes[0]->normal(), planes[1]->normal()));
            } 

            bool orient_h;
            float orient_v;
            if (dot(plane_h->normal(), vec3(0, 1, 0)) > 0)
                orient_h = true;
            else
                orient_h = false;
            orient_v = plane_v->orient(vec3(0, 0, 0)) ;
                
            LOG(INFO) << " orient_v: " << orient_v ;
            // dir = cross(l42.direction(), planes[i]->normal());

            std::vector<double> dist_h1, dist_h2;
            std::vector<vec3> points_v1, points_v2, points_h1, points_h2;
            std::vector<double> dist_v1, dist_v2;

            ok = planes[0]->intersect(*plane_h, line_h);
            ok = planes[0]->intersect(*plane_v, line_v);

            for (auto e : meshes[i]->edges()){
                const SurfaceMesh::Halfedge h0 = meshes[i]->halfedge(e, 0);
                const SurfaceMesh::Halfedge h1 = meshes[i]->halfedge(e, 1);

                const vec3 p0 = (vec3) meshes[i]->position(meshes[i]->target(h0));
                const vec3 p1 = (vec3) meshes[i]->position(meshes[i]->target(h1));

                bool parallel_h1 = std::abs(dot(Line3::from_two_points(p0, p1).direction(), line_h.direction())) > std::cos(theta);
                if (plane_h->orient(p0) >0 && parallel_h1){
                    float dist0 = line_h.squared_distance(p0);
                    float dist1 = line_h.squared_distance(p1);
                    LOG(INFO) << " parallel_h1: " << dist0 << "  " << dist1;
                    dist_h1.push_back(dist0);
                    dist_h1.push_back(dist1);
                    points_h1.push_back(p0);
                    points_h1.push_back(p1);
                }

                bool parallel_h2 = std::abs(dot(Line3::from_two_points(p0, p1).direction(), line_h.direction())) > std::cos(theta);
                if (plane_h->orient(p0) <0 && parallel_h2){
                    float dist0 = line_h.squared_distance(p0);
                    float dist1 = line_h.squared_distance(p1);
                    LOG(INFO) << " parallel_h2: " << dist0 << "  " << dist1;
                    dist_h2.push_back(dist0);
                    dist_h2.push_back(dist1);
                    points_h2.push_back(p0);
                    points_h2.push_back(p1);
                }

                bool parallel_v = std::abs(dot(Line3::from_two_points(p0, p1).direction(), line_v.direction())) > std::cos(theta);
                if (plane_v->orient(p0) >0 && parallel_v){
                    float dist0 = line_v.squared_distance(p0);
                    float dist1 = line_v.squared_distance(p1);
                    LOG(INFO) << " parallel_v1: " << dist0 << "  " << dist1;
                    dist_v1.push_back(dist0);
                    dist_v1.push_back(dist1);
                    points_v1.push_back(p0);
                    points_v1.push_back(p1);
                } else if (plane_v->orient(p0) <0 && parallel_v) {
                    float dist0 = line_v.squared_distance(p0);
                    float dist1 = line_v.squared_distance(p1);
                    LOG(INFO) << " parallel_v2: " << dist0 << "  " << dist1;
                    dist_v2.push_back(dist0);
                    dist_v2.push_back(dist1);
                    points_v2.push_back(p0);
                    points_v2.push_back(p1);
                }

            }  // end of edges

            std::vector<double>::iterator min_h1 = std::min_element(dist_h1.begin(), dist_h1.end());
            if (dist_h1.empty()) {
                LOG(ERROR) << "dist_h1.empty ";
                return ;
            }
            LOG(INFO) << "min_h1: " << sqrt(*min_h1);
            std::vector<double>::iterator min_h2 = std::min_element(dist_h2.begin(), dist_h2.end());
            if (dist_h2.empty()) {
                LOG(ERROR) << "dist_h2.empty ";
                return ;
            }
            LOG(INFO) << "min_h2: " << sqrt(*min_h2);
            LOG(INFO) << "min_h: " << sqrt(*min_h2) + sqrt(*min_h1);

            if (orient_h == false) {
                line_left.set(points_h1[std::distance(dist_h1.begin(), min_h1)], line_h.direction());
                line_right.set(points_h2[std::distance(dist_h2.begin(), min_h2)], line_h.direction());
            } else {
                line_right.set(points_h1[std::distance(dist_h1.begin(), min_h1)], line_h.direction());
                line_left.set(points_h2[std::distance(dist_h2.begin(), min_h2)], line_h.direction());
            }


            std::vector<double>::iterator min_v1 = std::min_element(dist_v1.begin(), dist_v1.end());
            if (dist_v1.empty()) {
                LOG(ERROR) << "dist_v1.empty ";
                return ;
            }
            LOG(INFO) << "min_v1: " << sqrt(*min_v1);
            std::vector<double>::iterator min_v2 = std::min_element(dist_v2.begin(), dist_v2.end());
            if (dist_v2.empty()) {
                LOG(ERROR) << "dist_v2.empty ";
                return ;
            }
            LOG(INFO) << "min_v2: " << sqrt(*min_v2);
            LOG(INFO) << "min_v: " << sqrt(*min_v2) + sqrt(*min_v1);

            if (orient_v > 0) {
                line_bottom.set(points_v1[std::distance(dist_v1.begin(), min_v1)], line_v.direction());
                plane_bottom = new Plane3(points_v1[std::distance(dist_v1.begin(), min_v1)], plane_v->normal());
                line_top.set(points_v2[std::distance(dist_v2.begin(), min_v2)], line_v.direction());
                plane_top = new Plane3(points_v2[std::distance(dist_v2.begin(), min_v2)], plane_v->normal());
            } else {
                line_top.set(points_v1[std::distance(dist_v1.begin(), min_v1)], line_v.direction());
                plane_top = new Plane3(points_v1[std::distance(dist_v1.begin(), min_v1)], plane_v->normal());
                line_bottom.set(points_v2[std::distance(dist_v2.begin(), min_v2)], line_v.direction());
                plane_bottom = new Plane3(points_v2[std::distance(dist_v2.begin(), min_v2)], plane_v->normal());
            }

            vec3 temp;

            if (plane_top->intersect(line_left, temp))
                interpoint.push_back(temp); 
            if (plane_top->intersect(line_right, temp))
                interpoint.push_back(temp);
            if (plane_bottom->intersect(line_right, temp))
                interpoint.push_back(temp);
            if (plane_bottom->intersect(line_left, temp))
                interpoint.push_back(temp);

        } // end of each mesh

        Plane3 *plane_center_v;
        vec3 point_top, point_bottom;
        vec3 point_mid_top, point_mid_bottom;
        double angle_center_v;
        plane_center_v = new Plane3(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 0, 1));
        if (plane_center_v->intersect(line_top, point_top) && plane_center_v->intersect(line_bottom, point_bottom))
            LOG(INFO) << "plane_center_v: sucess";
        angle_center_v = acos(std::abs(dot(normalize(point_top - point_bottom), line_left.direction())));
        angle_center_v = angle_center_v * 180.0 /3.1415926 ;
        LOG(INFO) << "angle_center_v: " << angle_center_v;
        // LOG(INFO) << "distance: " << easy3d::distance(interpoint[0], interpoint[1]);
        float del1 = sqrt(line_left.squared_distance(point_top));
        float del2 = sqrt(line_left.squared_distance(point_bottom));
        // LOG(INFO) << del1 << "  " << del2;
        if (del1 > del2)
            angle_center_v *= -1;
        point_mid_top = (interpoint[0] + interpoint[1]) / 2.0;
        point_mid_bottom = (interpoint[2] + interpoint[3]) / 2.0;

        // calculate last
        std::stringstream stream;
        stream << "rect info: "     << std::endl;
        stream << std::fixed << std::setprecision(3) ;
        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "left_top: " << "(" << interpoint[0] << ")"<< std::endl;
        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "right_top: " << "(" << interpoint[1] << ")"<< std::endl;
        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "right_bottom: " << "(" << interpoint[2] << ")"<< std::endl;
        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "left_bottom: " << "(" << interpoint[3] << ")"<< std::endl;

        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "length: " << easy3d::distance(interpoint[0], interpoint[3]) << "m"<< std::endl;
        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "width: "  << easy3d::distance(interpoint[0], interpoint[1]) << "m"<< std::endl;
        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "barrier height: "  << 2*sqrt(height_squared) << "m"<< std::endl;
        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "distane of lidar to plane: " << sqrt(planes[0]->squared_distance(vec3(0,0,0))) << "m"<< std::endl;
        stream << std::fixed << std::setprecision(2) ;
        stream << std::setw(14)<< " " << "deflection angle from center_line: " << angle_center_v << "degree"<< std::endl;
        LOG(WARNING) << stream.str();

        std::vector<Graph::Vertex> g_edges;
        std::vector<Graph::Vertex> center_v_edges;
        Graph* outcome = new Graph; 
        Graph* center_v = new Graph; 
        outcome->set_name("outcome");
        center_v->set_name("center_v");
        for (std::size_t i = 0; i < interpoint.size(); i++) {
            g_edges.push_back(outcome->add_vertex(interpoint[i]));
        }
        g_edges.push_back(outcome->add_vertex(point_mid_top));
        g_edges.push_back(outcome->add_vertex(point_mid_bottom));

        center_v_edges.push_back(center_v->add_vertex(point_top));
        center_v_edges.push_back(center_v->add_vertex(point_bottom));

        if (interpoint.size() == 4) {
            outcome->add_edge(g_edges[0], g_edges[1]);
            outcome->add_edge(g_edges[1], g_edges[2]);
            outcome->add_edge(g_edges[2], g_edges[3]);
            outcome->add_edge(g_edges[3], g_edges[0]);
        }
        outcome->add_edge(g_edges[4], g_edges[5]);
        center_v->add_edge(center_v_edges[0], center_v_edges[1]);

        viewer_->addModel(outcome);
        viewer_->addModel(center_v);

        // color it manual
        auto edges = outcome->renderer()->get_lines_drawable("edges");
        // edges->set_line_width(5.0f);
        edges->set_uniform_coloring(vec4(0, 1, 0, 1));

        auto vertices = outcome->renderer()->get_points_drawable("vertices");
        vertices->set_uniform_coloring(vec4(1, 0, 0, 1));
        vertices->set_point_size(10.0f);
        // vertices->set_visible(false);

        updateUi();
        viewer_->update();

    }
}

void MainWindow::show_box() {

    const float theta = static_cast<float>(3.1415926 * 3.0 / 180.0f);  

    auto my_cloud = dynamic_cast<PointCloud*>(viewer_->currentModel());
    if (my_cloud)
    {

        ::internal::EnrichedPointCloud point_set(my_cloud, my_cloud->vertex_property<int>("v:primitive_index"));

        auto hypothesis_ = new ::internal::Hypothesis;
        hypothesis_->generate(point_set);

        auto &planes = hypothesis_->supporting_planes();
        LOG(INFO) << planes.size() << " planes generated";

        Line3 l42, line_h, line_v;
        Line3 line_bottom, line_left, line_right, line_top;
   
        bool ok;
        ok = planes[0]->intersect(*planes[1], l42);
        LOG(INFO) << "l42 ok: " << ok;

        // calculate height
        // auto v_height_z = my_cloud->vertex_property<float>("v:height_plane");
        // for (auto v : my_cloud->vertices()) {
        //     const auto& p = my_cloud->position(v);
        //     float d = planes[0]->squared_distance(p);
        //     if (d > 0.49) d = 0.49;
        //     if (planes[0]->orient(p) < 0)
        //         d *= -1;
        //         v_height_z[v] = d;
        // }
        // LOG(INFO) << "calculate distance to plane done. ";

        auto &meshes = hypothesis_->supporting_meshes();
        if (meshes.size() != planes.size()){
            LOG(INFO) << "meshes.size() != planes.size()";
            return;
        }

        auto &centers = hypothesis_->supporting_planes_center();

        // float d0_squared = planes[1]->squared_distance(centers[4]);
        // LOG(INFO) << "bbox width: " << sqrt(d0_squared);

        // Plane3 *plane_yoz;
        // plane_yoz = new Plane3(vec3(0, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1));
        // for (std::size_t i = 1; i < meshes.size(); i++) {
        //     LOG(INFO) << "plane_yoz_orient: " << plane_yoz->orient(centers[i]);
        // }

        std::vector<vec3> interpoint;
        float height_squared = 0;

        for (std::size_t i = 2; i < 4; i++) {
            float height_temp = planes[0]->squared_distance(centers[i]);
            if (height_squared < height_temp)
                height_squared = height_temp;
        }
        LOG(INFO) << "LanGao: " << 2*sqrt(height_squared);

        for (std::size_t i = 1; i < meshes.size(); i++) {
            SurfaceMeshPolygonization polygonizer;
            polygonizer.apply(meshes[i]);
            viewer_->addModel(meshes[i]);
        }

        for (std::size_t i = 0; i < 1; i++) {
            SurfaceMeshPolygonization polygonizer;
            polygonizer.apply(meshes[i]);
            viewer_->addModel(meshes[i]);

            std::vector<vec3> interpoint_v, interpoint_h;
            
            Plane3* plane_v;
            Plane3* plane_h;
            Plane3* plane_bottom;
            Plane3* plane_top;

            plane_v = new Plane3(centers[i], planes[1]->normal());
            plane_h = new Plane3(centers[i], cross(planes[0]->normal(), planes[1]->normal()));

            bool orient_h;
            float orient_v;
            if (dot(plane_h->normal(), vec3(0, 1, 0)) > 0)
                orient_h = true;
            else
                orient_h = false;
            orient_v = plane_v->orient(vec3(0, 0, 0)) ;
                
            LOG(INFO) << " orient_v: " << orient_v ;
            // dir = cross(l42.direction(), planes[i]->normal());

            std::vector<double> dist_h1, dist_h2;
            std::vector<vec3> points_v1, points_v2, points_h1, points_h2;
            std::vector<double> dist_v1, dist_v2;

            ok = planes[0]->intersect(*plane_h, line_h);
            ok = planes[0]->intersect(*plane_v, line_v);

            for (auto e : meshes[i]->edges()){
                const SurfaceMesh::Halfedge h0 = meshes[i]->halfedge(e, 0);
                const SurfaceMesh::Halfedge h1 = meshes[i]->halfedge(e, 1);

                const vec3 p0 = (vec3) meshes[i]->position(meshes[i]->target(h0));
                const vec3 p1 = (vec3) meshes[i]->position(meshes[i]->target(h1));

                bool parallel_h1 = std::abs(dot(Line3::from_two_points(p0, p1).direction(), line_h.direction())) > std::cos(theta);
                if (plane_h->orient(p0) >0 && parallel_h1){
                    float dist0 = line_h.squared_distance(p0);
                    float dist1 = line_h.squared_distance(p1);
                    LOG(INFO) << " parallel_h1: " << dist0 << "  " << dist1;
                    dist_h1.push_back(dist0);
                    dist_h1.push_back(dist1);
                    points_h1.push_back(p0);
                    points_h1.push_back(p1);
                }

                bool parallel_h2 = std::abs(dot(Line3::from_two_points(p0, p1).direction(), line_h.direction())) > std::cos(theta);
                if (plane_h->orient(p0) <0 && parallel_h2){
                    float dist0 = line_h.squared_distance(p0);
                    float dist1 = line_h.squared_distance(p1);
                    LOG(INFO) << " parallel_h2: " << dist0 << "  " << dist1;
                    dist_h2.push_back(dist0);
                    dist_h2.push_back(dist1);
                    points_h2.push_back(p0);
                    points_h2.push_back(p1);
                }

                bool parallel_v = std::abs(dot(Line3::from_two_points(p0, p1).direction(), line_v.direction())) > std::cos(theta);
                if (plane_v->orient(p0) >0 && parallel_v){
                    float dist0 = line_v.squared_distance(p0);
                    float dist1 = line_v.squared_distance(p1);
                    LOG(INFO) << " parallel_v1: " << dist0 << "  " << dist1;
                    dist_v1.push_back(dist0);
                    dist_v1.push_back(dist1);
                    points_v1.push_back(p0);
                    points_v1.push_back(p1);
                } else if (plane_v->orient(p0) <0 && parallel_v) {
                    float dist0 = line_v.squared_distance(p0);
                    float dist1 = line_v.squared_distance(p1);
                    LOG(INFO) << " parallel_v2: " << dist0 << "  " << dist1;
                    dist_v2.push_back(dist0);
                    dist_v2.push_back(dist1);
                    points_v2.push_back(p0);
                    points_v2.push_back(p1);
                }

            }  // end of edges

            std::vector<double>::iterator min_h1 = std::min_element(dist_h1.begin(), dist_h1.end());
            if (dist_h1.empty()) {
                LOG(ERROR) << "dist_h1.empty ";
                return ;
            }
            LOG(INFO) << "min_h1: " << sqrt(*min_h1);
            std::vector<double>::iterator min_h2 = std::min_element(dist_h2.begin(), dist_h2.end());
            if (dist_h2.empty()) {
                LOG(ERROR) << "dist_h2.empty ";
                return ;
            }
            LOG(INFO) << "min_h2: " << sqrt(*min_h2);
            LOG(INFO) << "min_h: " << sqrt(*min_h2) + sqrt(*min_h1);

            if (orient_h == false) {
                line_left.set(points_h1[std::distance(dist_h1.begin(), min_h1)], line_h.direction());
                line_right.set(points_h2[std::distance(dist_h2.begin(), min_h2)], line_h.direction());
            } else {
                line_right.set(points_h1[std::distance(dist_h1.begin(), min_h1)], line_h.direction());
                line_left.set(points_h2[std::distance(dist_h2.begin(), min_h2)], line_h.direction());
            }


            std::vector<double>::iterator min_v1 = std::min_element(dist_v1.begin(), dist_v1.end());
            if (dist_v1.empty()) {
                LOG(ERROR) << "dist_v1.empty ";
                return ;
            }
            LOG(INFO) << "min_v1: " << sqrt(*min_v1);
            std::vector<double>::iterator min_v2 = std::min_element(dist_v2.begin(), dist_v2.end());
            if (dist_v2.empty()) {
                LOG(ERROR) << "dist_v2.empty ";
                return ;
            }
            LOG(INFO) << "min_v2: " << sqrt(*min_v2);
            LOG(INFO) << "min_v: " << sqrt(*min_v2) + sqrt(*min_v1);

            if (orient_v > 0) {
                line_bottom.set(points_v1[std::distance(dist_v1.begin(), min_v1)], line_v.direction());
                plane_bottom = new Plane3(points_v1[std::distance(dist_v1.begin(), min_v1)], plane_v->normal());
                line_top.set(points_v2[std::distance(dist_v2.begin(), min_v2)], line_v.direction());
                plane_top = new Plane3(points_v2[std::distance(dist_v2.begin(), min_v2)], plane_v->normal());
            } else {
                line_top.set(points_v1[std::distance(dist_v1.begin(), min_v1)], line_v.direction());
                plane_top = new Plane3(points_v1[std::distance(dist_v1.begin(), min_v1)], plane_v->normal());
                line_bottom.set(points_v2[std::distance(dist_v2.begin(), min_v2)], line_v.direction());
                plane_bottom = new Plane3(points_v2[std::distance(dist_v2.begin(), min_v2)], plane_v->normal());
            }

            vec3 temp;

            if (plane_top->intersect(line_left, temp))
                interpoint.push_back(temp); 
            if (plane_top->intersect(line_right, temp))
                interpoint.push_back(temp);
            if (plane_bottom->intersect(line_right, temp))
                interpoint.push_back(temp);
            if (plane_bottom->intersect(line_left, temp))
                interpoint.push_back(temp);

        } // end of each mesh

        Plane3 *plane_center_v;
        vec3 point_top, point_bottom;
        vec3 point_mid_top, point_mid_bottom;
        double angle_center_v;
        plane_center_v = new Plane3(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 0, 1));
        if (plane_center_v->intersect(line_top, point_top) && plane_center_v->intersect(line_bottom, point_bottom))
            LOG(INFO) << "plane_center_v: sucess";
        angle_center_v = acos(std::abs(dot(normalize(point_top - point_bottom), line_left.direction())));
        angle_center_v = angle_center_v * 180.0 /3.1415926 ;
        LOG(INFO) << "angle_center_v: " << angle_center_v;
        // LOG(INFO) << "distance: " << easy3d::distance(interpoint[0], interpoint[1]);
        float del1 = sqrt(line_left.squared_distance(point_top));
        float del2 = sqrt(line_left.squared_distance(point_bottom));
        // LOG(INFO) << del1 << "  " << del2;
        if (del1 > del2)
            angle_center_v *= -1;
        point_mid_top = (interpoint[0] + interpoint[1]) / 2.0;
        point_mid_bottom = (interpoint[2] + interpoint[3]) / 2.0;

        // calculate last
        std::stringstream stream;
        stream << "rect info: "     << std::endl;
        stream << std::fixed << std::setprecision(3) ;
        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "left_top: " << "(" << interpoint[0] << ")"<< std::endl;
        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "right_top: " << "(" << interpoint[1] << ")"<< std::endl;
        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "right_bottom: " << "(" << interpoint[2] << ")"<< std::endl;
        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "left_bottom: " << "(" << interpoint[3] << ")"<< std::endl;

        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "length: " << easy3d::distance(interpoint[0], interpoint[3]) << "m"<< std::endl;
        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "width: "  << easy3d::distance(interpoint[0], interpoint[1]) << "m"<< std::endl;
        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "barrier height: "  << 2*sqrt(height_squared) << "m"<< std::endl;
        stream << std::setw(14)<< " " << std::left << std::setw(1)<< "distane of lidar to plane: " << sqrt(planes[0]->squared_distance(vec3(0,0,0))) << "m"<< std::endl;
        stream << std::fixed << std::setprecision(2) ;
        stream << std::setw(14)<< " " << "deflection angle from center_line: " << angle_center_v << "degree"<< std::endl;
        LOG(WARNING) << stream.str();

        std::vector<Graph::Vertex> g_edges;
        std::vector<Graph::Vertex> center_v_edges;
        Graph* outcome = new Graph; 
        Graph* center_v = new Graph; 
        outcome->set_name("outcome");
        center_v->set_name("center_v");
        for (std::size_t i = 0; i < interpoint.size(); i++) {
            g_edges.push_back(outcome->add_vertex(interpoint[i]));
        }
        g_edges.push_back(outcome->add_vertex(point_mid_top));
        g_edges.push_back(outcome->add_vertex(point_mid_bottom));

        center_v_edges.push_back(center_v->add_vertex(point_top));
        center_v_edges.push_back(center_v->add_vertex(point_bottom));

        if (interpoint.size() == 4) {
            outcome->add_edge(g_edges[0], g_edges[1]);
            outcome->add_edge(g_edges[1], g_edges[2]);
            outcome->add_edge(g_edges[2], g_edges[3]);
            outcome->add_edge(g_edges[3], g_edges[0]);
        }
        outcome->add_edge(g_edges[4], g_edges[5]);
        center_v->add_edge(center_v_edges[0], center_v_edges[1]);

        viewer_->addModel(outcome);
        viewer_->addModel(center_v);

        // color it manual
        auto edges = outcome->renderer()->get_lines_drawable("edges");
        // edges->set_line_width(5.0f);
        edges->set_uniform_coloring(vec4(0, 1, 0, 1));

        auto vertices = outcome->renderer()->get_points_drawable("vertices");
        vertices->set_uniform_coloring(vec4(1, 0, 0, 1));
        vertices->set_point_size(10.0f);
        // vertices->set_visible(false);

        updateUi();
        viewer_->update();

    }
}

void MainWindow::show_onekey() {

    Yaml::Node params_;
    Yaml::Parse (params_, "./params.yaml");
    std::vector<float> normal_ground_f;
    std::vector<float> normal_top_f;
    std::vector<float> plane_front_f;
    std::vector<float> plane_front_up_f;
    std::vector<float> plane_center_v_f;
    Line3 rotate_line;
    vec3 rotate_o;
    // As float, double, bool, int
    bool projection = params_["projection"].As<bool> ();
    bool cgal = params_["cgal"].As<bool> ();
    bool grid = params_["grid"].As<bool> ();
    bool use_leftri = params_["use_leftri"].As<bool> ();
    float theta_h_f = params_["theta_h"].As<float> ();
    float theta_v_f = params_["theta_v"].As<float> ();
    float theta_normal_f = params_["theta_normal"].As<float> ();
    float borders_d1 = params_["borders_d1"].As<float> ();
    float borders_d2 = params_["borders_d2"].As<float> ();
    float alpha_spacing = params_["alpha_spacing"].As<float> ();
    float tail_d1 = params_["tail_d1"].As<float> ();
    float sample_threshold = params_["sample_threshold"].As<float> ();
    float cgal_line_threshold = params_["cgal_line_threshold"].As<float> ();
    int k_least = params_["k_least"].As<int> ();
    std::string normal_ground_s = params_["normal_ground"].As<std::string> ();
    easy3d::string::split(normal_ground_s, ' ', normal_ground_f);
    std::string normal_top_s = params_["normal_top"].As<std::string> ();
    easy3d::string::split(normal_top_s, ' ', normal_top_f);
    std::string plane_front_s = params_["plane_front"].As<std::string> ();
    easy3d::string::split(plane_front_s, ' ', plane_front_f);
    std::string plane_front_up_s = params_["plane_front_up"].As<std::string> ();
    easy3d::string::split(plane_front_up_s, ' ', plane_front_up_f);

    std::string plane_center_v_s = params_["plane_center_v"].As<std::string> ();
    easy3d::string::split(plane_center_v_s, ' ', plane_center_v_f);

    std::string rotate_line_s = params_["rotate_line"].As<std::string> ();
    std::stringstream rotate_line_stream(rotate_line_s);
    rotate_line_stream >> rotate_line;
    // LOG(INFO) << a; 
    // LOG(INFO) << b; 
    // return ;

    const float theta_h          = geom::to_radians(theta_h_f);  
    const float theta_v          = geom::to_radians(theta_v_f); 
    const float theta_normal     = geom::to_radians(theta_normal_f); 
    const float theta_normal_max = geom::to_radians(30.0); 

    const vec3  normal_ground    = vec3(plane_front_up_f[3], plane_front_up_f[4], plane_front_up_f[5]); 

    const vec3  normal_top       = vec3(plane_front_f[3], plane_front_f[4], plane_front_f[5]); 
    // const vec3  normal_left      = cross(normal_ground, normal_top); 
    const vec3  normal_left      = vec3(plane_center_v_f[3], plane_center_v_f[4], plane_center_v_f[5]); 

    // const Plane3 *plane_center_v = new Plane3(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 0, 1));
    const Plane3 *plane_center_v    = new Plane3(vec3(plane_center_v_f[0], plane_center_v_f[1], plane_center_v_f[2]),
                                    vec3(plane_center_v_f[3], plane_center_v_f[4], plane_center_v_f[5]) );

    const Plane3 *plane_front    = new Plane3(vec3(plane_front_f[0], plane_front_f[1], plane_front_f[2]),
                                    vec3(plane_front_f[3], plane_front_f[4], plane_front_f[5]) );

    const Plane3 *plane_origin   = new Plane3(vec3(0), normal_ground);
    const Plane3 *plane_front_up   = new Plane3(vec3(plane_front_up_f[0], plane_front_up_f[1], plane_front_up_f[2]), 
                                                        normal_ground);
#if 0
    if (!plane_origin->intersect(rotate_line, rotate_o))
        LOG(ERROR) << "plane_origin->intersect(rotate_line, rotate_o)";
#else
    rotate_o = plane_origin->projection(plane_center_v->projection(vec3(0)));
#endif

    std::vector<Plane_Center> planes_updown;
    std::vector<Plane_Center> planes_topbot;
    std::vector<Plane_Center> planes_leftri;

    std::vector<vec3> alpha_points;
    std::vector<unsigned int> alpha_indices;
    std::vector<vec3> interpoint;
    std::vector<vec3> rect;
    std::vector<vec3> cgal_rect;
    cgal_rect.resize(6);
    std::vector<vec3> cgal_rect_project;
    cgal_rect_project.resize(6);
    vec3 point_top, point_bottom;
    vec3 cgal_point_top, cgal_point_bottom;
    vec3 point_mid_top, point_mid_bottom;
    float box_width;
    float dist_center_ransac;
    float height_squared = 999.0f;
    float dis_tail_min = 0;
    float dis_tail_max = 0;
    float dis_tail_average = 0;
    double angle_center_v;
    double cgal_angle_center_v;
    float cgal_dist_rotate_o;

    auto my_cloud = dynamic_cast<PointCloud*>(viewer_->currentModel());
    if (my_cloud)
    {
        ::internal::down_sample(my_cloud, sample_threshold, k_least, grid);

        ::internal::EnrichedPointCloud point_set(my_cloud, my_cloud->vertex_property<int>("v:primitive_index"));

        auto hypothesis_ = new ::internal::Hypothesis;
        hypothesis_->generate(point_set);

        auto &planes = hypothesis_->supporting_planes();
        LOG(WARNING) << planes.size() << " planes generated";
        

        auto &centers = hypothesis_->supporting_planes_center();
        auto &segs = hypothesis_->supporting_segments();
        if (segs.size() != planes.size()){
            LOG(WARNING) << "segs.size() != planes.size()";
            delete hypothesis_;
            return;
        }

        planes_updown.clear();
        planes_topbot.clear();
        planes_leftri.clear();

        for (std::size_t i = 0; i < planes.size(); i++) {
            struct Plane_Center temp;
            temp.plane = planes[i];
            temp.center = centers[i];
            temp.mesh = nullptr;
            temp.seg = segs[i];
            if (geom::angle2(planes[i]->normal(), normal_ground) < theta_normal) {
                planes_updown.push_back(temp);
            } else
            if (geom::angle2(planes[i]->normal(), normal_top) < theta_normal) {
                planes_topbot.push_back(temp);
            } else 
            if (geom::angle2(planes[i]->normal(), normal_left) < theta_normal) {
                planes_leftri.push_back(temp);
            }
        }
        const int size_updown = planes_updown.size();
        LOG(WARNING) << "planes_updown size: " << size_updown;
        std::sort(planes_updown.begin(), planes_updown.end(), ::internal::OriginDistanceIncreasing<Plane_Center>());

        const int size_topbot = planes_topbot.size();
        LOG(WARNING) << "planes_topbot size: " << size_topbot;
        std::sort(planes_topbot.begin(), planes_topbot.end(), ::internal::OriginDistanceIncreasing<Plane_Center>());

        const int size_leftri = planes_leftri.size();
        LOG(WARNING) << "planes_leftri size: " << size_leftri;
        // std::sort(planes_leftri.begin(), planes_leftri.end(), ::internal::OriginDistanceIncreasing<Plane_Center>());
        if (size_leftri >=2)
        {
            vec3 s(0, 0, 0);
            vec3 t(10000, 0, 0);
            vec3 p;
            bool k = planes_leftri[0].plane->intersect(s, t, p); 
            if (k)
                std::swap(planes_leftri[0], planes_leftri[1]);
        }

        if (size_updown >=2 && size_topbot >=1) {
            // calculate LanGao
            for (std::size_t i = 0; i < planes_leftri.size(); i++) {
                float height_temp = planes_updown[0].plane->squared_distance(planes_leftri[i].center);
                if (height_squared > height_temp)
                    height_squared = height_temp;
            }

            if (size_leftri == 2) {
                LOG(WARNING) << "width using planes: " << easy3d::distance(planes_leftri[0].center, planes_leftri[1].center);
            }

            // StopWatch w;
            // w.start();
            // planes_updown[0].mesh = planes_updown[0].seg->borders_plus(borders_d1, borders_d2, plane_front);
            planes_updown[0].mesh = planes_updown[0].seg->borders_alpha_plus(borders_d1, borders_d2, plane_front, alpha_spacing);
            // LOG(WARNING) << "done. " << w.time_string();

            // SurfaceMeshPolygonization polygonizer;
            // polygonizer.apply(planes_updown[0].mesh);
            // viewer_->addModel(planes_updown[0].mesh);
            // see alpha shape
            // auto alpha_mesh = planes_updown[0].seg->borders_alpha();
            // viewer_->addModel(alpha_mesh);
            // auto models = viewer_->models();
            // for (auto m : models) {
            //     if (m->name() == "Plane_plus") {
            //         viewer_->deleteModel(m);
            //     }
            // }
            // viewer_->addModel(planes_updown[0].mesh);
            // updateUi();
///////////////////////////^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
            std::vector<vec3> interpoint_v, interpoint_h;
            std::vector<Point_3>  cgal_points_h1;
            std::vector<Point_3>  cgal_points_h2;
            std::vector<Point_3>  cgal_points_v1;
            std::vector<Point_3>  cgal_points_v2;

            
            Plane3* plane_v;
            Plane3* plane_h;
            Plane3* plane_bottom;
            Plane3* plane_top;
            Plane3* cgal_plane_left = nullptr;
            Plane3* cgal_plane_right = nullptr;
            Plane3* cgal_plane_centerv = nullptr;
            Plane3* cgal_plane_bottom = nullptr;
            Line3 line_h, line_v;
            Line3 line_bottom, line_left, line_right, line_top;
            bool ok;

            plane_v = new Plane3(planes_updown[0].center, planes_topbot[size_topbot -1].plane->normal());
            plane_h = new Plane3(planes_updown[0].center, 
                                cross(planes_updown[0].plane->normal(), planes_topbot[size_topbot -1].plane->normal()));

            bool orient_h;
            float orient_v;
            if (dot(plane_h->normal(), vec3(1, 0, 0)) > 0)
                orient_h = true;
            else
                orient_h = false;
            orient_v = plane_v->orient(vec3(0, 0, 0)) ;
                
            std::vector<double> dist_h1, dist_h2;
            std::vector<vec3> points_v1, points_v2, points_h1, points_h2;
            std::vector<double> dist_v1, dist_v2;

            ok = planes_updown[0].plane->intersect(*plane_h, line_h);
            ok = planes_updown[0].plane->intersect(*plane_v, line_v);

            auto &this_mesh = planes_updown[0].mesh;
            auto &this_seg = planes_updown[0].seg;
            auto &this_center = planes_updown[0].center;

            // auto this_borders = this_mesh->renderer()->get_lines_drawable("borders");
            // this_borders->set_line_width(2.0f);
            // this_borders->set_uniform_coloring(vec4(1, 0, 0, 1));   
            // this_borders->set_visible(true);
            // auto this_faces = this_mesh->renderer()->get_triangles_drawable("faces");
            // this_faces->set_visible(false);
            // auto this_vertices = this_mesh->renderer()->get_points_drawable("vertices");
            // this_vertices->set_uniform_coloring(vec4(1, 1, 0, 0.7));
            // this_vertices->set_point_size(4.0f);
            // this_vertices->set_visible(true);

            // this_mesh->renderer()->update();

// StopWatch w;
// w.start();
            for (auto e : this_mesh->edges()){
                if (this_mesh->is_border(e)) {
                const SurfaceMesh::Halfedge h0 = this_mesh->halfedge(e, 0);
                const SurfaceMesh::Halfedge h1 = this_mesh->halfedge(e, 1);

                const vec3 p0 = (vec3) this_mesh->position(this_mesh->target(h0));
                const vec3 p1 = (vec3) this_mesh->position(this_mesh->target(h1));

                alpha_points.push_back(p0);
                alpha_points.push_back(p1);

                bool parallel_h = geom::angle2(p0 - p1, line_h.direction()) < theta_h;
                if (plane_h->orient(p0) >0 && parallel_h){
                    float dist0 = line_h.squared_distance(p0);
                    float dist1 = line_h.squared_distance(p1);
                    dist_h1.push_back(dist0);
                    dist_h1.push_back(dist1);
                    points_h1.push_back(p0);
                    points_h1.push_back(p1);
                    cgal_points_h1.push_back(Point_3(p0.x, p0.y, p0.z));
                    cgal_points_h1.push_back(Point_3(p1.x, p1.y, p1.z));
                } else if (plane_h->orient(p0) <0 && parallel_h) {
                    float dist0 = line_h.squared_distance(p0);
                    float dist1 = line_h.squared_distance(p1);
                    dist_h2.push_back(dist0);
                    dist_h2.push_back(dist1);
                    points_h2.push_back(p0);
                    points_h2.push_back(p1);
                    cgal_points_h2.push_back(Point_3(p0.x, p0.y, p0.z));
                    cgal_points_h2.push_back(Point_3(p1.x, p1.y, p1.z));
                }


                bool parallel_v = geom::angle2(p0 - p1, line_v.direction()) < theta_v;
                if (plane_v->orient(p0) >0 && parallel_v){
                    float dist0 = line_v.squared_distance(p0);
                    float dist1 = line_v.squared_distance(p1);
                    dist_v1.push_back(dist0);
                    dist_v1.push_back(dist1);
                    points_v1.push_back(p0);
                    points_v1.push_back(p1);
                    cgal_points_v1.push_back(Point_3(p0.x, p0.y, p0.z));
                    cgal_points_v1.push_back(Point_3(p1.x, p1.y, p1.z));
                } else if (plane_v->orient(p0) <0 && parallel_v) {
                    float dist0 = line_v.squared_distance(p0);
                    float dist1 = line_v.squared_distance(p1);
                    dist_v2.push_back(dist0);
                    dist_v2.push_back(dist1);
                    points_v2.push_back(p0);
                    points_v2.push_back(p1);
                    cgal_points_v2.push_back(Point_3(p0.x, p0.y, p0.z));
                    cgal_points_v2.push_back(Point_3(p1.x, p1.y, p1.z));
                }
                }   // end if (this_mesh->is_border(e)) 
            }  // end of edges
// LOG(WARNING) << "is_border done. " << w.time_string();
            std::vector<double>::iterator min_h1 = std::min_element(dist_h1.begin(), dist_h1.end());
            if (dist_h1.empty()) {
                LOG(ERROR) << "dist_h1.empty ";
                return ;
            }
            // LOG(INFO) << "min_h1: " << sqrt(*min_h1);
            std::vector<double>::iterator min_h2 = std::min_element(dist_h2.begin(), dist_h2.end());
            if (dist_h2.empty()) {
                LOG(ERROR) << "dist_h2.empty ";
                return ;
            }
            // LOG(INFO) << "min_h2: " << sqrt(*min_h2);
            // LOG(INFO) << "min_h: " << sqrt(*min_h2) + sqrt(*min_h1);

            if (orient_h == false) {
                line_left.set(points_h1[std::distance(dist_h1.begin(), min_h1)], line_h.direction());
                line_right.set(points_h2[std::distance(dist_h2.begin(), min_h2)], line_h.direction());
            } else {
                line_right.set(points_h1[std::distance(dist_h1.begin(), min_h1)], line_h.direction());
                line_left.set(points_h2[std::distance(dist_h2.begin(), min_h2)], line_h.direction());
            }
            // new cgal methord for h
            Line_3 cgal_line_h1;
            Line_3 cgal_line_h2;
            Line_3 cgal_line_left, cgal_line_right;
            Vector_3 cgal_direction_left, cgal_direction_right;
            vec3 direction_left, direction_right;
            Line3 easy3d_line_left, easy3d_line_right;
            
            CGAL::linear_least_squares_fitting_3(cgal_points_h1.begin(), cgal_points_h1.end(), cgal_line_h1, CGAL::Dimension_tag<0>());
            CGAL::linear_least_squares_fitting_3(cgal_points_h2.begin(), cgal_points_h2.end(), cgal_line_h2, CGAL::Dimension_tag<0>());
            std::vector<Point_3>::iterator it = cgal_points_h1.begin();
            while (it != cgal_points_h1.end()) {
                if (CGAL::squared_distance(*it, cgal_line_h1) > cgal_line_threshold * cgal_line_threshold)
                    it = cgal_points_h1.erase(it);
                else
                    it++;
            }
            it = cgal_points_h2.begin();
            while (it != cgal_points_h2.end()) {
                if (CGAL::squared_distance(*it, cgal_line_h2) > cgal_line_threshold * cgal_line_threshold)
                    it = cgal_points_h2.erase(it);
                else
                    it++;
            }
            CGAL::linear_least_squares_fitting_3(cgal_points_h1.begin(), cgal_points_h1.end(), cgal_line_h1, CGAL::Dimension_tag<0>());
            CGAL::linear_least_squares_fitting_3(cgal_points_h2.begin(), cgal_points_h2.end(), cgal_line_h2, CGAL::Dimension_tag<0>());

            if (orient_h == true) {
                cgal_line_right = cgal_line_h1;
                cgal_line_left = cgal_line_h2;
            }
            else {
                cgal_line_right = cgal_line_h2;
                cgal_line_left = cgal_line_h1;
            }
            cgal_direction_left = cgal_line_left.to_vector();
            cgal_direction_right = cgal_line_right.to_vector();
            auto cgal_point_line_left = vec3(cgal_line_left.point().x(),
                                              cgal_line_left.point().y(),
                                              cgal_line_left.point().z());
            auto cgal_point_line_right = vec3(cgal_line_right.point().x(),
                                              cgal_line_right.point().y(),
                                              cgal_line_right.point().z());
            auto cgal_dir_line_left = vec3(cgal_direction_left.x(),
                                              cgal_direction_left.y(),
                                              cgal_direction_left.z());
            auto cgal_dir_line_right = vec3(cgal_direction_right.x(),
                                              cgal_direction_right.y(),
                                              cgal_direction_right.z());
            easy3d_line_left.set(cgal_point_line_left, cgal_dir_line_left);
            easy3d_line_right.set(cgal_point_line_right, cgal_dir_line_right);

        if (size_leftri == 2 && use_leftri) {

            cgal_plane_left = planes_leftri[0].plane;
            cgal_plane_right = planes_leftri[1].plane;

            Line3 cgal_plane_intersect;
            vec3 cgal_centerv_plane_normal;
            bool okk;
            okk = cgal_plane_left->intersect(*cgal_plane_right, cgal_plane_intersect);
            if (okk) {
                float cgal_thetra = geom::to_degrees(geom::angle(cgal_plane_left->normal(), cgal_plane_right->normal()));

                if (cgal_thetra > 90)
                    cgal_centerv_plane_normal = -cgal_plane_left->normal() + cgal_plane_right->normal();
                else
                    cgal_centerv_plane_normal = cgal_plane_left->normal() + cgal_plane_right->normal();
                cgal_centerv_plane_normal.normalize();
                cgal_plane_centerv = new Plane3(cgal_plane_intersect.point(), cgal_centerv_plane_normal);
            } else {
                auto center_right = planes_leftri[1].center;
                auto center = cgal_plane_left->projection(center_right);
                center= (center + center_right) * 0.5f;
                cgal_plane_centerv = new Plane3(center, cgal_plane_left->normal());
            }
        
        } else {    

            cgal_plane_left = new Plane3(cgal_point_line_left, 
                                cross(planes_updown[0].plane->normal(), cgal_dir_line_left));
            cgal_plane_right = new Plane3(cgal_point_line_right, 
                                cross(planes_updown[0].plane->normal(), cgal_dir_line_right));

            Line3 cgal_plane_intersect;
            vec3 cgal_centerv_plane_normal;
            bool okk;
            okk = cgal_plane_left->intersect(*cgal_plane_right, cgal_plane_intersect);
            if (okk) {
                float cgal_thetra = geom::to_degrees(geom::angle(cgal_plane_left->normal(), cgal_plane_right->normal()));

                if (cgal_thetra > 90)
                    cgal_centerv_plane_normal = -cgal_plane_left->normal() + cgal_plane_right->normal();
                else
                    cgal_centerv_plane_normal = cgal_plane_left->normal() + cgal_plane_right->normal();
                cgal_centerv_plane_normal.normalize();
                cgal_plane_centerv = new Plane3(cgal_plane_intersect.point(), cgal_centerv_plane_normal);
            } else {
                auto center = cgal_plane_left->projection(cgal_point_line_right);
                center= (center + cgal_point_line_right) * 0.5f;
                cgal_plane_centerv = new Plane3(center, cgal_plane_left->normal());
            }
        }
//////////// vvvvvvvvvvvvvvvvvvvvv ///////////////////////////////////////////////////////////////////
            std::vector<double>::iterator min_v1 = std::min_element(dist_v1.begin(), dist_v1.end());
            if (dist_v1.empty()) {
                LOG(ERROR) << "dist_v1.empty ";
                return ;
            }
            // LOG(INFO) << "min_v1: " << sqrt(*min_v1);
            std::vector<double>::iterator min_v2 = std::min_element(dist_v2.begin(), dist_v2.end());
            if (dist_v2.empty()) {
                LOG(ERROR) << "dist_v2.empty ";
                return ;
            }

            if (orient_v > 0) {
                line_bottom.set(points_v1[std::distance(dist_v1.begin(), min_v1)], line_v.direction());
                plane_bottom = new Plane3(points_v1[std::distance(dist_v1.begin(), min_v1)], plane_v->normal());
                line_top.set(points_v2[std::distance(dist_v2.begin(), min_v2)], line_v.direction());
                plane_top = new Plane3(points_v2[std::distance(dist_v2.begin(), min_v2)], plane_v->normal());
            } else {
                line_top.set(points_v1[std::distance(dist_v1.begin(), min_v1)], line_v.direction());
                plane_top = new Plane3(points_v1[std::distance(dist_v1.begin(), min_v1)], plane_v->normal());
                line_bottom.set(points_v2[std::distance(dist_v2.begin(), min_v2)], line_v.direction());
                plane_bottom = new Plane3(points_v2[std::distance(dist_v2.begin(), min_v2)], plane_v->normal());
            }
            // new cgal methord for v
            Line_3 cgal_line_v1;
            Line_3 cgal_line_v2;
            Line_3 cgal_line_top, cgal_line_bottom;
            Vector_3 cgal_direction_top, cgal_direction_bottom;
            vec3 direction_top, direction_bottom;
            Line3 easy3d_line_top, easy3d_line_bottom;
            CGAL::linear_least_squares_fitting_3(cgal_points_v1.begin(), cgal_points_v1.end(), cgal_line_v1, CGAL::Dimension_tag<0>());
            CGAL::linear_least_squares_fitting_3(cgal_points_v2.begin(), cgal_points_v2.end(), cgal_line_v2, CGAL::Dimension_tag<0>());
            it = cgal_points_v1.begin();
            while (it != cgal_points_v1.end()) {
                if (CGAL::squared_distance(*it, cgal_line_v1) > cgal_line_threshold * cgal_line_threshold)
                    it = cgal_points_v1.erase(it);
                else
                    it++;
            }
            it = cgal_points_v2.begin();
            while (it != cgal_points_v2.end()) {
                if (CGAL::squared_distance(*it, cgal_line_v2) > cgal_line_threshold * cgal_line_threshold)
                    it = cgal_points_v2.erase(it);
                else
                    it++;
            }
            CGAL::linear_least_squares_fitting_3(cgal_points_v1.begin(), cgal_points_v1.end(), cgal_line_v1, CGAL::Dimension_tag<0>());
            CGAL::linear_least_squares_fitting_3(cgal_points_v2.begin(), cgal_points_v2.end(), cgal_line_v2, CGAL::Dimension_tag<0>());

            if (orient_v > 0) {
                cgal_line_bottom = cgal_line_v1;
                cgal_line_top = cgal_line_v2;
            }
            else {
                cgal_line_bottom = cgal_line_v2;
                cgal_line_top = cgal_line_v1;
            }
            cgal_direction_top = cgal_line_top.to_vector();
            cgal_direction_bottom = cgal_line_bottom.to_vector();
            auto cgal_point_line_top = vec3(cgal_line_top.point().x(),
                                              cgal_line_top.point().y(),
                                              cgal_line_top.point().z());
            auto cgal_point_line_bottom = vec3(cgal_line_bottom.point().x(),
                                              cgal_line_bottom.point().y(),
                                              cgal_line_bottom.point().z());
            auto cgal_dir_line_top = vec3(cgal_direction_top.x(),
                                              cgal_direction_top.y(),
                                              cgal_direction_top.z());
            auto cgal_dir_line_bottom = vec3(cgal_direction_bottom.x(),
                                              cgal_direction_bottom.y(),
                                              cgal_direction_bottom.z());
            easy3d_line_top.set(cgal_point_line_top, cgal_dir_line_top);
            easy3d_line_bottom.set(cgal_point_line_bottom, cgal_dir_line_bottom);
            // 6 points to projection
            if (!cgal_plane_left->intersect(easy3d_line_top, cgal_rect[0]))
                LOG(ERROR) << "cgal_rect[0]";
            if (!cgal_plane_right->intersect(easy3d_line_top, cgal_rect[1]))
                LOG(ERROR) << "cgal_rect[1]";
            if (!cgal_plane_right->intersect(easy3d_line_bottom, cgal_rect[2]))
                LOG(ERROR) << "cgal_rect[2]";
            if (!cgal_plane_left->intersect(easy3d_line_bottom, cgal_rect[3]))
                LOG(ERROR) << "cgal_rect[3]";
            if (!cgal_plane_centerv->intersect(easy3d_line_top, cgal_rect[4]))
                LOG(ERROR) << "cgal_rect[4]";
            if (!cgal_plane_centerv->intersect(easy3d_line_bottom, cgal_rect[5]))
                LOG(ERROR) << "cgal_rect[5]";
            // new width calculation test
            Plane3 *cgal_plane_centerv_oth = nullptr;
            vec3 cgal_point_centerv_oth_left;
            vec3 cgal_point_centerv_oth_right;
            cgal_plane_centerv_oth = new Plane3(this_center, cgal_rect[4] - cgal_rect[5]);
            cgal_plane_centerv_oth->intersect(easy3d_line_left, cgal_point_centerv_oth_left);
            cgal_plane_centerv_oth->intersect(easy3d_line_right, cgal_point_centerv_oth_right);
            LOG(WARNING) << "cgal_width: " << easy3d::distance(cgal_point_centerv_oth_left, cgal_point_centerv_oth_right);
            if (cgal_plane_centerv_oth) 
                delete cgal_plane_centerv_oth;

            // projection
            if (projection) {
                for (int i = 0; i < cgal_rect.size(); i++)
                    cgal_rect_project[i] = plane_origin->projection(cgal_rect[i]);
            }
            // tail distance
            cgal_plane_bottom = new Plane3(cgal_rect[5], cgal_rect[4] - cgal_rect[5]);
            this_seg->distance_tail(cgal_plane_bottom, plane_origin, tail_d1, dis_tail_min, dis_tail_max, dis_tail_average);
            // hypothesis_->refine_plane_sub(this_seg, plane_bottom, plane_origin, tail_d1, dis_tail_min, dis_tail_max, dis_tail_average);
            // angle and bias
            if (projection)
                if (plane_center_v->intersect(Line3::from_two_points(cgal_rect_project[0], cgal_rect_project[1]), cgal_point_top) && 
                plane_center_v->intersect(Line3::from_two_points(cgal_rect_project[2], cgal_rect_project[3]), cgal_point_bottom))
                LOG(INFO) << "cgal plane_center_v: sucess";
            else
                if (plane_center_v->intersect(Line3::from_two_points(cgal_rect[0], cgal_rect[1]), cgal_point_top) && 
                plane_center_v->intersect(Line3::from_two_points(cgal_rect[2], cgal_rect[3]), cgal_point_bottom))
                LOG(INFO) << "cgal plane_center_v: sucess";
            if (projection) {
                vec3 clock_wise = cross(cgal_rect_project[0] - cgal_rect_project[5], cgal_rect_project[1] - cgal_rect_project[5]);
                vec3 clock_cgal = cross(cgal_point_top - cgal_point_bottom, cgal_rect_project[4] - cgal_rect_project[5]);
                cgal_angle_center_v = geom::to_degrees(geom::angle2((cgal_point_top - cgal_point_bottom), 
                    (cgal_rect_project[4] - cgal_rect_project[5])));
                if (cgal_angle_center_v > 1e-3) {
                    if (geom::to_degrees(geom::angle(clock_wise, clock_cgal)) > 90)
                        cgal_angle_center_v *= -1;        
                }
            } else {
                vec3 clock_wise = cross(cgal_rect[0] - cgal_rect[5], cgal_rect[1] - cgal_rect[5]);
                vec3 clock_cgal = cross(cgal_point_top - cgal_point_bottom, cgal_rect[4] - cgal_rect[5]);
                cgal_angle_center_v = geom::to_degrees(geom::angle2((cgal_point_top - cgal_point_bottom), 
                    (cgal_rect[4] - cgal_rect[5])));
                if (cgal_angle_center_v > 1e-3) {
                    if (geom::to_degrees(geom::angle(clock_wise, clock_cgal)) > 90)
                        cgal_angle_center_v *= -1;        
                }
            }
            if (projection) {
                vec3 clock_wise = cross(cgal_rect_project[0] - cgal_rect_project[5], cgal_rect_project[1] - cgal_rect_project[5]);
                vec3 clock_cgal = cross(rotate_o - cgal_rect_project[5], cgal_rect_project[4] - cgal_rect_project[5]);

                cgal_dist_rotate_o = Line3::from_two_points(cgal_rect_project[4], cgal_rect_project[5]).squared_distance(rotate_o);
                cgal_dist_rotate_o = sqrt(cgal_dist_rotate_o);
                if (cgal_dist_rotate_o > 1e-3) {
                    if (geom::to_degrees(geom::angle(clock_wise, clock_cgal)) > 90)
                        cgal_dist_rotate_o *= -1;        
                }
                // LOG(INFO) << "cgal_dist_rotate_o: " << cgal_dist_rotate_o;
            }
            else {
                vec3 clock_wise = cross(cgal_rect[0] - cgal_rect[5], cgal_rect[1] - cgal_rect[5]);
                vec3 clock_cgal = cross(rotate_o - cgal_rect[5], cgal_rect[4] - cgal_rect[5]);

                cgal_dist_rotate_o = Line3::from_two_points(cgal_rect[4], cgal_rect[5]).squared_distance(rotate_o);
                cgal_dist_rotate_o = sqrt(cgal_dist_rotate_o);
                if (cgal_dist_rotate_o > 1e-3) {
                    if (geom::to_degrees(geom::angle(clock_wise, clock_cgal)) > 90)
                        cgal_dist_rotate_o *= -1;        
                }
                // LOG(INFO) << "cgal_dist_rotate_o: " << cgal_dist_rotate_o;
            }
            // distane of bottom to plane_front
            vec3 car_bottom_middle;
            car_bottom_middle = (cgal_rect[2] + cgal_rect[3]) * 0.5f;
            // length
            easy3d::distance(cgal_rect_project[4], cgal_rect_project[5]);
            // width
            (easy3d::distance(cgal_rect_project[0], cgal_rect_project[1]) +
             easy3d::distance(cgal_rect_project[2], cgal_rect_project[3])) * 0.5;
//////////////////////////////////////////////////////// end of v ////////////////////////////////////////////////////
            vec3 temp;
            Line3 choose_line_left, choose_line_right;
            if (cgal) {
                choose_line_left = easy3d_line_left;
                choose_line_right = easy3d_line_right;
            } else {
                choose_line_left = line_left;
                choose_line_right = line_right;
    
            }

            if (projection) {
                if (plane_top->intersect(choose_line_left, temp))
                    interpoint.push_back(plane_origin->projection(temp)); 
                if (plane_top->intersect(choose_line_right, temp))
                    interpoint.push_back(plane_origin->projection(temp));
                if (plane_bottom->intersect(choose_line_right, temp))
                    interpoint.push_back(plane_origin->projection(temp));
                if (plane_bottom->intersect(choose_line_left, temp))
                    interpoint.push_back(plane_origin->projection(temp));
            } else {

                if (plane_top->intersect(choose_line_left, temp))
                    interpoint.push_back(temp); 
                if (plane_top->intersect(choose_line_right, temp))
                    interpoint.push_back(temp);
                if (plane_bottom->intersect(choose_line_right, temp))
                    interpoint.push_back(temp);
                if (plane_bottom->intersect(choose_line_left, temp))
                    interpoint.push_back(temp);
            }

            if (plane_top->intersect(choose_line_left, temp))
                rect.push_back(temp); 
            if (plane_top->intersect(choose_line_right, temp))
                rect.push_back(temp);
            if (plane_bottom->intersect(choose_line_right, temp))
                rect.push_back(temp);
            if (plane_bottom->intersect(choose_line_left, temp))
                rect.push_back(temp);

            
            if (plane_center_v->intersect(Line3::from_two_points(interpoint[0], interpoint[1]), point_top) && 
                plane_center_v->intersect(Line3::from_two_points(interpoint[2], interpoint[3]), point_bottom))
                LOG(INFO) << "plane_center_v: sucess";
            
            angle_center_v = geom::angle2( (point_top - point_bottom), (interpoint[0] - interpoint[3]) );
            angle_center_v = geom::to_degrees(angle_center_v);
            // LOG(INFO) << "angle_center_v: " << angle_center_v;

            float del1 = distance2(point_top, interpoint[0]);
            float del2 = distance2(point_bottom, interpoint[3]);
            if (del1 > del2)
                angle_center_v *= -1;
            point_mid_top = (interpoint[0] + interpoint[1]) / 2.0;
            point_mid_bottom = (interpoint[2] + interpoint[3]) / 2.0;

            // this_seg->distance_tail(plane_bottom, plane_origin, tail_d1, dis_tail_max, dis_tail_average);

            float sdist_rotate_o = Line3::from_two_points(point_mid_top, point_mid_bottom).squared_distance(rotate_o);
            sdist_rotate_o = sqrt(sdist_rotate_o);
            // LOG(INFO) << "sdist_rotate_o: " << sdist_rotate_o;
#if 0
            // calculate last
            std::stringstream stream;
            stream << "rect info: "     << std::endl;
            stream << std::fixed << std::setprecision(3) ;
            stream << std::setw(14)<< " " << std::left << std::setw(1)<< "left_top: " << "(" << rect[0] << ")"<< std::endl;
            stream << std::setw(14)<< " " << std::left << std::setw(1)<< "right_top: " << "(" << rect[1] << ")"<< std::endl;
            stream << std::setw(14)<< " " << std::left << std::setw(1)<< "right_bottom: " << "(" << rect[2] << ")"<< std::endl;
            stream << std::setw(14)<< " " << std::left << std::setw(1)<< "left_bottom: " << "(" << rect[3] << ")"<< std::endl;

            stream << std::setw(14)<< " " << std::left << std::setw(1)<< "length: " << easy3d::distance(interpoint[0], interpoint[3]) << "m"<< std::endl;
            stream << std::setw(14)<< " " << std::left << std::setw(1)<< "width: "  << easy3d::distance(interpoint[0], interpoint[1]) << "m"<< std::endl;
            stream << std::setw(14)<< " " << std::left << std::setw(1)<< "barrier height: "  << 2*sqrt(height_squared) << "m"<< std::endl;
            stream << std::setw(14)<< " " << std::left << std::setw(1)<< "distane of lidar to plane_front_up: " 
                    << sqrt(plane_front_up->squared_distance(vec3(0,0,0))) << "m"<< std::endl;
            stream << std::setw(14)<< " " << std::left << std::setw(1)<< "distane of lidar to plane_ransac: " 
                    << sqrt(planes_updown[0].plane->squared_distance(vec3(0,0,0))) << "m"<< std::endl;
            stream << std::setw(14)<< " " << std::left << std::setw(1)<< "distane of bottom to plane_front: " 
                    << sqrt(plane_front->squared_distance((rect[3] + rect[2]) * 0.5f)) << "m"<< std::endl;
            stream << std::setw(14)<< " " << std::left << std::setw(1)<< "distane of tail to plane_origin: "
                    <<dis_tail_min << " " << dis_tail_max << " " << dis_tail_average << "m" << std::endl;
            // stream << std::fixed << std::setprecision(2) ;
            stream << std::setw(14)<< " " << "deflection angle from center_line: " << angle_center_v << " degree"<< std::endl;
            stream << std::setw(14)<< " " << "deflection distance from center_line: " << sdist_rotate_o << " m"<< std::endl;
            LOG(WARNING) << stream.str();
#endif
            // cgal calculate last
            std::stringstream cgal_stream;
            cgal_stream << "cgal_rect info: "     << std::endl;
            cgal_stream << std::fixed << std::setprecision(3) ;
            cgal_stream << std::setw(14)<< " " << std::left << std::setw(1)<< "left_top: " << "(" << cgal_rect[0] << ")"<< std::endl;
            cgal_stream << std::setw(14)<< " " << std::left << std::setw(1)<< "right_top: " << "(" << cgal_rect[1] << ")"<< std::endl;
            cgal_stream << std::setw(14)<< " " << std::left << std::setw(1)<< "right_bottom: " << "(" << cgal_rect[2] << ")"<< std::endl;
            cgal_stream << std::setw(14)<< " " << std::left << std::setw(1)<< "left_bottom: " << "(" << cgal_rect[3] << ")"<< std::endl;

            cgal_stream << std::setw(14)<< " " << std::left << std::setw(1)<< "length: " << easy3d::distance(cgal_rect_project[4], cgal_rect_project[5]) << " m"<< std::endl;
            cgal_stream << std::setw(14)<< " " << std::left << std::setw(1)<< "width: "  << easy3d::distance(cgal_rect_project[0], cgal_rect_project[1]) << "->"
                                                                                            << easy3d::distance(cgal_rect_project[2], cgal_rect_project[3])  << "->"
                                                                                            << 0.5f * (easy3d::distance(cgal_rect_project[0], cgal_rect_project[1]) +
                                                                                                       easy3d::distance(cgal_rect_project[2], cgal_rect_project[3]) ) 
                                                                                            << " m"<< std::endl;
            cgal_stream << std::setw(14)<< " " << std::left << std::setw(1)<< "barrier height: "  << 2*sqrt(height_squared) << " m"<< std::endl;
            cgal_stream << std::setw(14)<< " " << std::left << std::setw(1)<< "distane of lidar to plane_front_up: " 
                    << sqrt(plane_front_up->squared_distance(vec3(0,0,0))) << " m"<< std::endl;
            cgal_stream << std::setw(14)<< " " << std::left << std::setw(1)<< "distane of lidar to plane_ransac: " 
                    << sqrt(planes_updown[0].plane->squared_distance(vec3(0,0,0))) << " m"<< std::endl;
            cgal_stream << std::setw(14)<< " " << std::left << std::setw(1)<< "distane of plane_ransac center to plane_origin: " 
                    << sqrt(plane_origin->squared_distance(this_center)) << " m"<< std::endl;
            cgal_stream << std::setw(14)<< " " << std::left << std::setw(1)<< "distane of bottom to plane_front: " 
                    << sqrt(plane_front->squared_distance(car_bottom_middle)) << " m"<< std::endl;
            cgal_stream << std::setw(14)<< " " << std::left << std::setw(1)<< "distane of tail to plane_origin: "
                    <<dis_tail_min << " " << dis_tail_max << " " << dis_tail_average << " m" << std::endl;
            // cgal_stream << std::fixed << std::setprecision(2) ;
            cgal_stream << std::setw(14)<< " " << "deflection angle from center_line: " << cgal_angle_center_v << " degree"<< std::endl;
            cgal_stream << std::setw(14)<< " " << "deflection distance from center_line: " << cgal_dist_rotate_o << " m"<< std::endl;
            LOG(WARNING) << cgal_stream.str();
            // clean
            delete plane_v;
            delete plane_h;
            delete plane_bottom;
            delete plane_top;

        } else
        // size_topbot size_leftri
        if (size_updown >=2 && size_topbot ==0 && size_leftri ==2) {

            // planes_updown[1].mesh = planes_updown[1].seg->borders_plus(borders_d1, borders_d2, plane_front);
            planes_updown[1].mesh = planes_updown[1].seg->borders_alpha_plus(borders_d1, borders_d2, plane_front, alpha_spacing);

            // SurfaceMeshPolygonization polygonizer;
            // polygonizer.apply(planes_updown[1].mesh);

            // auto models = viewer_->models();
            // for (auto m : models) {
            //     if (m->name() == "Plane_plus") {
            //         viewer_->deleteModel(m);
            //     }
            // }
            // viewer_->addModel(planes_updown[1].mesh);
            // updateUi();

            std::vector<Point_3>  cgal_points_h1;
            std::vector<Point_3>  cgal_points_h2;
            std::vector<Point_3>  cgal_points_v1;
            std::vector<Point_3>  cgal_points_v2;
            
            Plane3* plane_v;
            Plane3* plane_h;
            Plane3* cgal_plane_left = nullptr;
            Plane3* cgal_plane_right = nullptr;
            Plane3* cgal_plane_centerv = nullptr;
            Plane3* cgal_plane_bottom = nullptr;
            Line3 line_h, line_v;
            bool ok;

            plane_h = new Plane3(planes_updown[1].center, planes_leftri[size_leftri -1].plane->normal());
            plane_v = new Plane3(planes_updown[1].center, 
                                cross(planes_updown[1].plane->normal(), planes_leftri[size_leftri -1].plane->normal()));

            bool orient_h;
            float orient_v;
            if (dot(plane_h->normal(), vec3(1, 0, 0)) > 0)
                orient_h = true;
            else
                orient_h = false;
            orient_v = plane_v->orient(vec3(0, 0, 0)) ;


            ok = planes_updown[1].plane->intersect(*plane_h, line_h);
            ok = planes_updown[1].plane->intersect(*plane_v, line_v);

            auto &this_mesh = planes_updown[1].mesh;
            auto &this_seg = planes_updown[1].seg;
            auto &this_center = planes_updown[1].center;


            for (auto e : this_mesh->edges()){
                if (this_mesh->is_border(e)) {
                const SurfaceMesh::Halfedge h0 = this_mesh->halfedge(e, 0);
                const SurfaceMesh::Halfedge h1 = this_mesh->halfedge(e, 1);

                const vec3 p0 = (vec3) this_mesh->position(this_mesh->target(h0));
                const vec3 p1 = (vec3) this_mesh->position(this_mesh->target(h1));

                alpha_points.push_back(p0);
                alpha_points.push_back(p1);

                bool parallel_h = geom::angle2(p0 - p1, line_h.direction()) < theta_h;
                if (plane_h->orient(p0) >0 && parallel_h){
                    // float dist0 = line_h.squared_distance(p0);
                    // float dist1 = line_h.squared_distance(p1);
                    // dist_h1.push_back(dist0);
                    // dist_h1.push_back(dist1);
                    // points_h1.push_back(p0);
                    // points_h1.push_back(p1);
                    cgal_points_h1.push_back(Point_3(p0.x, p0.y, p0.z));
                    cgal_points_h1.push_back(Point_3(p1.x, p1.y, p1.z));
                } else if (plane_h->orient(p0) <0 && parallel_h) {
                    // float dist0 = line_h.squared_distance(p0);
                    // float dist1 = line_h.squared_distance(p1);
                    // dist_h2.push_back(dist0);
                    // dist_h2.push_back(dist1);
                    // points_h2.push_back(p0);
                    // points_h2.push_back(p1);
                    cgal_points_h2.push_back(Point_3(p0.x, p0.y, p0.z));
                    cgal_points_h2.push_back(Point_3(p1.x, p1.y, p1.z));
                }


                bool parallel_v = geom::angle2(p0 - p1, line_v.direction()) < theta_v;
                if (plane_v->orient(p0) >0 && parallel_v){
                    // float dist0 = line_v.squared_distance(p0);
                    // float dist1 = line_v.squared_distance(p1);
                    // dist_v1.push_back(dist0);
                    // dist_v1.push_back(dist1);
                    // points_v1.push_back(p0);
                    // points_v1.push_back(p1);
                    cgal_points_v1.push_back(Point_3(p0.x, p0.y, p0.z));
                    cgal_points_v1.push_back(Point_3(p1.x, p1.y, p1.z));
                } else if (plane_v->orient(p0) <0 && parallel_v) {
                    // float dist0 = line_v.squared_distance(p0);
                    // float dist1 = line_v.squared_distance(p1);
                    // dist_v2.push_back(dist0);
                    // dist_v2.push_back(dist1);
                    // points_v2.push_back(p0);
                    // points_v2.push_back(p1);
                    cgal_points_v2.push_back(Point_3(p0.x, p0.y, p0.z));
                    cgal_points_v2.push_back(Point_3(p1.x, p1.y, p1.z));
                }
                }   // end if (this_mesh->is_border(e)) 
            }  // end of edges

            // new cgal methord for h
            Line_3 cgal_line_h1;
            Line_3 cgal_line_h2;
            Line_3 cgal_line_left, cgal_line_right;
            Vector_3 cgal_direction_left, cgal_direction_right;
            vec3 direction_left, direction_right;
            Line3 easy3d_line_left, easy3d_line_right;
            
            CGAL::linear_least_squares_fitting_3(cgal_points_h1.begin(), cgal_points_h1.end(), cgal_line_h1, CGAL::Dimension_tag<0>());
            CGAL::linear_least_squares_fitting_3(cgal_points_h2.begin(), cgal_points_h2.end(), cgal_line_h2, CGAL::Dimension_tag<0>());
            std::vector<Point_3>::iterator it = cgal_points_h1.begin();
            while (it != cgal_points_h1.end()) {
                if (CGAL::squared_distance(*it, cgal_line_h1) > cgal_line_threshold * cgal_line_threshold)
                    it = cgal_points_h1.erase(it);
                else
                    it++;
            }
            it = cgal_points_h2.begin();
            while (it != cgal_points_h2.end()) {
                if (CGAL::squared_distance(*it, cgal_line_h2) > cgal_line_threshold * cgal_line_threshold)
                    it = cgal_points_h2.erase(it);
                else
                    it++;
            }
            CGAL::linear_least_squares_fitting_3(cgal_points_h1.begin(), cgal_points_h1.end(), cgal_line_h1, CGAL::Dimension_tag<0>());
            CGAL::linear_least_squares_fitting_3(cgal_points_h2.begin(), cgal_points_h2.end(), cgal_line_h2, CGAL::Dimension_tag<0>());

            if (orient_h == true) {
                cgal_line_right = cgal_line_h1;
                cgal_line_left = cgal_line_h2;
            }
            else {
                cgal_line_right = cgal_line_h2;
                cgal_line_left = cgal_line_h1;
            }
            cgal_direction_left = cgal_line_left.to_vector();
            cgal_direction_right = cgal_line_right.to_vector();
            auto cgal_point_line_left = vec3(cgal_line_left.point().x(),
                                              cgal_line_left.point().y(),
                                              cgal_line_left.point().z());
            auto cgal_point_line_right = vec3(cgal_line_right.point().x(),
                                              cgal_line_right.point().y(),
                                              cgal_line_right.point().z());
            auto cgal_dir_line_left = vec3(cgal_direction_left.x(),
                                              cgal_direction_left.y(),
                                              cgal_direction_left.z());
            auto cgal_dir_line_right = vec3(cgal_direction_right.x(),
                                              cgal_direction_right.y(),
                                              cgal_direction_right.z());
            easy3d_line_left.set(cgal_point_line_left, cgal_dir_line_left);
            easy3d_line_right.set(cgal_point_line_right, cgal_dir_line_right);

            cgal_plane_left = planes_leftri[0].plane;
            cgal_plane_right = planes_leftri[1].plane;

            Line3 cgal_plane_intersect;
            vec3 cgal_centerv_plane_normal;
            bool okk;
            okk = cgal_plane_left->intersect(*cgal_plane_right, cgal_plane_intersect);
            if (okk) {
                float cgal_thetra = geom::to_degrees(geom::angle(cgal_plane_left->normal(), cgal_plane_right->normal()));

                if (cgal_thetra > 90)
                    cgal_centerv_plane_normal = -cgal_plane_left->normal() + cgal_plane_right->normal();
                else
                    cgal_centerv_plane_normal = cgal_plane_left->normal() + cgal_plane_right->normal();
                cgal_centerv_plane_normal.normalize();
                cgal_plane_centerv = new Plane3(cgal_plane_intersect.point(), cgal_centerv_plane_normal);
            } else {
                auto center_right = planes_leftri[1].center;
                auto center = cgal_plane_left->projection(center_right);
                center= (center + center_right) * 0.5f;
                cgal_plane_centerv = new Plane3(center, cgal_plane_left->normal());
            }

            // new cgal methord for v
            Line_3 cgal_line_v1;
            Line_3 cgal_line_v2;
            Line_3 cgal_line_top, cgal_line_bottom;
            Vector_3 cgal_direction_top, cgal_direction_bottom;
            vec3 direction_top, direction_bottom;
            Line3 easy3d_line_top, easy3d_line_bottom;
            CGAL::linear_least_squares_fitting_3(cgal_points_v1.begin(), cgal_points_v1.end(), cgal_line_v1, CGAL::Dimension_tag<0>());
            CGAL::linear_least_squares_fitting_3(cgal_points_v2.begin(), cgal_points_v2.end(), cgal_line_v2, CGAL::Dimension_tag<0>());
            it = cgal_points_v1.begin();
            while (it != cgal_points_v1.end()) {
                if (CGAL::squared_distance(*it, cgal_line_v1) > cgal_line_threshold * cgal_line_threshold)
                    it = cgal_points_v1.erase(it);
                else
                    it++;
            }
            it = cgal_points_v2.begin();
            while (it != cgal_points_v2.end()) {
                if (CGAL::squared_distance(*it, cgal_line_v2) > cgal_line_threshold * cgal_line_threshold)
                    it = cgal_points_v2.erase(it);
                else
                    it++;
            }
            CGAL::linear_least_squares_fitting_3(cgal_points_v1.begin(), cgal_points_v1.end(), cgal_line_v1, CGAL::Dimension_tag<0>());
            CGAL::linear_least_squares_fitting_3(cgal_points_v2.begin(), cgal_points_v2.end(), cgal_line_v2, CGAL::Dimension_tag<0>());

            if (orient_v > 0) {
                cgal_line_bottom = cgal_line_v1;
                cgal_line_top = cgal_line_v2;
            }
            else {
                cgal_line_bottom = cgal_line_v2;
                cgal_line_top = cgal_line_v1;
            }
            cgal_direction_top = cgal_line_top.to_vector();
            cgal_direction_bottom = cgal_line_bottom.to_vector();
            auto cgal_point_line_top = vec3(cgal_line_top.point().x(),
                                              cgal_line_top.point().y(),
                                              cgal_line_top.point().z());
            auto cgal_point_line_bottom = vec3(cgal_line_bottom.point().x(),
                                              cgal_line_bottom.point().y(),
                                              cgal_line_bottom.point().z());
            auto cgal_dir_line_top = vec3(cgal_direction_top.x(),
                                              cgal_direction_top.y(),
                                              cgal_direction_top.z());
            auto cgal_dir_line_bottom = vec3(cgal_direction_bottom.x(),
                                              cgal_direction_bottom.y(),
                                              cgal_direction_bottom.z());
            easy3d_line_top.set(cgal_point_line_top, cgal_dir_line_top);
            easy3d_line_bottom.set(cgal_point_line_bottom, cgal_dir_line_bottom);

            // 6 points to projection
            if (!cgal_plane_left->intersect(easy3d_line_top, cgal_rect[0]))
                LOG(ERROR) << "cgal_rect[0]";
            if (!cgal_plane_right->intersect(easy3d_line_top, cgal_rect[1]))
                LOG(ERROR) << "cgal_rect[1]";
            if (!cgal_plane_right->intersect(easy3d_line_bottom, cgal_rect[2]))
                LOG(ERROR) << "cgal_rect[2]";
            if (!cgal_plane_left->intersect(easy3d_line_bottom, cgal_rect[3]))
                LOG(ERROR) << "cgal_rect[3]";
            if (!cgal_plane_centerv->intersect(easy3d_line_top, cgal_rect[4]))
                LOG(ERROR) << "cgal_rect[4]";
            if (!cgal_plane_centerv->intersect(easy3d_line_bottom, cgal_rect[5]))
                LOG(ERROR) << "cgal_rect[5]";

            if (projection) {
                for (int i = 0; i < cgal_rect.size(); i++)
                    cgal_rect_project[i] = plane_origin->projection(cgal_rect[i]);
            }
            // center_ransac
            dist_center_ransac = sqrt(plane_origin->squared_distance(this_center));       

            // width
            { 
                vec3 temp0 = plane_origin->projection(planes_leftri[0].center);
                vec3 temp1 = plane_origin->projection(planes_leftri[1].center);
                box_width = easy3d::distance(temp0, temp1);
            }

            // tail distance
            cgal_plane_bottom = new Plane3(cgal_rect[5], cgal_rect[4] - cgal_rect[5]);
            this_seg->distance_tail(cgal_plane_bottom, plane_origin, tail_d1, dis_tail_min, dis_tail_max, dis_tail_average);
            // angle
            if (projection)
                if (plane_center_v->intersect(Line3::from_two_points(cgal_rect_project[0], cgal_rect_project[1]), cgal_point_top) && 
                plane_center_v->intersect(Line3::from_two_points(cgal_rect_project[2], cgal_rect_project[3]), cgal_point_bottom))
                LOG(INFO) << "cgal plane_center_v: sucess";
            else
                if (plane_center_v->intersect(Line3::from_two_points(cgal_rect[0], cgal_rect[1]), cgal_point_top) && 
                plane_center_v->intersect(Line3::from_two_points(cgal_rect[2], cgal_rect[3]), cgal_point_bottom))
                LOG(INFO) << "cgal plane_center_v: sucess";
            if (projection) {
                vec3 clock_wise = cross(cgal_rect_project[0] - cgal_rect_project[5], cgal_rect_project[1] - cgal_rect_project[5]);
                vec3 clock_cgal = cross(cgal_point_top - cgal_point_bottom, cgal_rect_project[4] - cgal_rect_project[5]);
                cgal_angle_center_v = geom::to_degrees(geom::angle2((cgal_point_top - cgal_point_bottom), 
                    (cgal_rect_project[4] - cgal_rect_project[5])));
                if (cgal_angle_center_v > 1e-3) {
                    if (geom::to_degrees(geom::angle(clock_wise, clock_cgal)) > 90)
                        cgal_angle_center_v *= -1;        
                }
            } else {
                vec3 clock_wise = cross(cgal_rect[0] - cgal_rect[5], cgal_rect[1] - cgal_rect[5]);
                vec3 clock_cgal = cross(cgal_point_top - cgal_point_bottom, cgal_rect[4] - cgal_rect[5]);
                cgal_angle_center_v = geom::to_degrees(geom::angle2((cgal_point_top - cgal_point_bottom), 
                    (cgal_rect[4] - cgal_rect[5])));
                if (cgal_angle_center_v > 1e-3) {
                    if (geom::to_degrees(geom::angle(clock_wise, clock_cgal)) > 90)
                        cgal_angle_center_v *= -1;        
                }
            }
            // bias
            if (projection) {
                vec3 clock_wise = cross(cgal_rect_project[0] - cgal_rect_project[5], cgal_rect_project[1] - cgal_rect_project[5]);
                vec3 clock_cgal = cross(rotate_o - cgal_rect_project[5], cgal_rect_project[4] - cgal_rect_project[5]);

                cgal_dist_rotate_o = Line3::from_two_points(cgal_rect_project[4], cgal_rect_project[5]).squared_distance(rotate_o);
                cgal_dist_rotate_o = sqrt(cgal_dist_rotate_o);
                if (cgal_dist_rotate_o > 1e-3) {
                    if (geom::to_degrees(geom::angle(clock_wise, clock_cgal)) > 90)
                        cgal_dist_rotate_o *= -1;        
                }
            }
            else {
                vec3 clock_wise = cross(cgal_rect[0] - cgal_rect[5], cgal_rect[1] - cgal_rect[5]);
                vec3 clock_cgal = cross(rotate_o - cgal_rect[5], cgal_rect[4] - cgal_rect[5]);

                cgal_dist_rotate_o = Line3::from_two_points(cgal_rect[4], cgal_rect[5]).squared_distance(rotate_o);
                cgal_dist_rotate_o = sqrt(cgal_dist_rotate_o);
                if (cgal_dist_rotate_o > 1e-3) {
                    if (geom::to_degrees(geom::angle(clock_wise, clock_cgal)) > 90)
                        cgal_dist_rotate_o *= -1;        
                }
            }
            // print last
            LOG(WARNING) << "box_width: " << box_width;
            LOG(WARNING) << "dist_center_ransac: " << dist_center_ransac;
            LOG(WARNING) << "dis_tail_min: " << dis_tail_min;
            LOG(WARNING) << "dis_tail_max: " << dis_tail_max;
            LOG(WARNING) << "dis_tail_average: " << dis_tail_average;
            LOG(WARNING) << "cgal_angle_center_v: " << cgal_angle_center_v;
            LOG(WARNING) << "cgal_dist_rotate_o: " << cgal_dist_rotate_o;

            // std::vector<double>::iterator min_h1 = std::min_element(dist_h1.begin(), dist_h1.end());
            // if (dist_h1.empty()) {
            //     LOG(ERROR) << "dist_h1.empty ";
            //     return ;
            // }
            // LOG(INFO) << "min_h1: " << sqrt(*min_h1);
            // std::vector<double>::iterator min_h2 = std::min_element(dist_h2.begin(), dist_h2.end());
            // if (dist_h2.empty()) {
            //     LOG(ERROR) << "dist_h2.empty ";
            //     return ;
            // }
            // LOG(INFO) << "min_h2: " << sqrt(*min_h2);

            // if (orient_h == false) {
            //     line_left.set(points_h1[std::distance(dist_h1.begin(), min_h1)], line_h.direction());
            //     line_right.set(points_h2[std::distance(dist_h2.begin(), min_h2)], line_h.direction());
            // } else {
            //     line_right.set(points_h1[std::distance(dist_h1.begin(), min_h1)], line_h.direction());
            //     line_left.set(points_h2[std::distance(dist_h2.begin(), min_h2)], line_h.direction());
            // }


            // std::vector<double>::iterator min_v1 = std::min_element(dist_v1.begin(), dist_v1.end());
            // if (dist_v1.empty()) {
            //     LOG(ERROR) << "dist_v1.empty ";
            //     return ;
            // }
            // LOG(INFO) << "min_v1: " << sqrt(*min_v1);
            // std::vector<double>::iterator min_v2 = std::min_element(dist_v2.begin(), dist_v2.end());
            // if (dist_v2.empty()) {
            //     LOG(ERROR) << "dist_v2.empty ";
            //     return ;
            // }
            // LOG(INFO) << "min_v2: " << sqrt(*min_v2);

            // if (orient_v > 0) {
            //     line_bottom.set(points_v1[std::distance(dist_v1.begin(), min_v1)], line_v.direction());
            //     plane_bottom = new Plane3(points_v1[std::distance(dist_v1.begin(), min_v1)], plane_v->normal());
            //     line_top.set(points_v2[std::distance(dist_v2.begin(), min_v2)], line_v.direction());
            //     plane_top = new Plane3(points_v2[std::distance(dist_v2.begin(), min_v2)], plane_v->normal());
            // } else {
            //     line_top.set(points_v1[std::distance(dist_v1.begin(), min_v1)], line_v.direction());
            //     plane_top = new Plane3(points_v1[std::distance(dist_v1.begin(), min_v1)], plane_v->normal());
            //     line_bottom.set(points_v2[std::distance(dist_v2.begin(), min_v2)], line_v.direction());
            //     plane_bottom = new Plane3(points_v2[std::distance(dist_v2.begin(), min_v2)], plane_v->normal());
            // }

            // vec3 temp;

            // if (projection) {
            //     if (plane_top->intersect(line_left, temp))
            //         interpoint.push_back(plane_origin->projection(temp)); 
            //     if (plane_top->intersect(line_right, temp))
            //         interpoint.push_back(plane_origin->projection(temp));
            //     if (plane_bottom->intersect(line_right, temp))
            //         interpoint.push_back(plane_origin->projection(temp));
            //     if (plane_bottom->intersect(line_left, temp))
            //         interpoint.push_back(plane_origin->projection(temp));
            // } else {

            //     if (plane_top->intersect(line_left, temp))
            //         interpoint.push_back(temp); 
            //     if (plane_top->intersect(line_right, temp))
            //         interpoint.push_back(temp);
            //     if (plane_bottom->intersect(line_right, temp))
            //         interpoint.push_back(temp);
            //     if (plane_bottom->intersect(line_left, temp))
            //         interpoint.push_back(temp);
            // }

            // if (plane_top->intersect(line_left, temp))
            //     rect.push_back(temp); 
            // if (plane_top->intersect(line_right, temp))
            //     rect.push_back(temp);
            // if (plane_bottom->intersect(line_right, temp))
            //     rect.push_back(temp);
            // if (plane_bottom->intersect(line_left, temp))
            //     rect.push_back(temp);

            // if (plane_center_v->intersect(Line3::from_two_points(interpoint[0], interpoint[1]), point_top) && 
            //     plane_center_v->intersect(Line3::from_two_points(interpoint[2], interpoint[3]), point_bottom))
            //     LOG(INFO) << "plane_center_v: sucess";
            
            // angle_center_v = geom::angle2( (point_top - point_bottom), (interpoint[0] - interpoint[3]) );
            // angle_center_v = geom::to_degrees(angle_center_v);
            // LOG(INFO) << "angle_center_v: " << angle_center_v;

            // float del1 = distance2(point_top, interpoint[0]);
            // float del2 = distance2(point_bottom, interpoint[3]);
            // if (del1 > del2)
            //     angle_center_v *= -1;
            // point_mid_top = (interpoint[0] + interpoint[1]) / 2.0;
            // point_mid_bottom = (interpoint[2] + interpoint[3]) / 2.0;


            // this_seg->distance_tail(plane_bottom, plane_origin, tail_d1, dis_tail_min, dis_tail_max, dis_tail_average);

            // float sdist_rotate_o = Line3::from_two_points(point_mid_top, point_mid_bottom).squared_distance(rotate_o);



            // std::stringstream stream;
            // stream << "rect info: "     << std::endl;
            // stream << std::fixed << std::setprecision(3) ;
            // // stream << std::setw(14)<< " " << std::left << std::setw(1)<< "left_top: " << "(" << rect[0] << ")"<< std::endl;
            // // stream << std::setw(14)<< " " << std::left << std::setw(1)<< "right_top: " << "(" << rect[1] << ")"<< std::endl;
            // stream << std::setw(14)<< " " << std::left << std::setw(1)<< "right_bottom: " << "(" << rect[2] << ")"<< std::endl;
            // stream << std::setw(14)<< " " << std::left << std::setw(1)<< "left_bottom: " << "(" << rect[3] << ")"<< std::endl;

            // // stream << std::setw(14)<< " " << std::left << std::setw(1)<< "length: " << easy3d::distance(interpoint[0], interpoint[3]) << "m"<< std::endl;
            // stream << std::setw(14)<< " " << std::left << std::setw(1)<< "width: "  << easy3d::distance(interpoint[0], interpoint[1]) << "m"<< std::endl;
            // stream << std::setw(14)<< " " << std::left << std::setw(1)<< "distane of lidar to plane_front_up: " 
            //         << sqrt(plane_front_up->squared_distance(vec3(0,0,0))) << "m"<< std::endl;
            // stream << std::setw(14)<< " " << std::left << std::setw(1)<< "distane of lidar to plane_up: " 
            //         << sqrt(planes_updown[0].plane->squared_distance(vec3(0,0,0))) << "m"<< std::endl;
            // stream << std::setw(14)<< " " << std::left << std::setw(1)<< "distane of lidar to plane_down: " 
            //         << sqrt(planes_updown[1].plane->squared_distance(vec3(0,0,0))) << "m"<< std::endl;
            // stream << std::setw(14)<< " " << std::left << std::setw(1)<< "distane of bottom to plane_front: " 
            //         << sqrt(plane_front->squared_distance((rect[3] + rect[2]) * 0.5f)) << "m"<< std::endl;
            // stream << std::setw(14)<< " " << std::left << std::setw(1)<< "distane of tail to plane_origin: "
            //         << dis_tail_max << " " << dis_tail_average << "m" << std::endl;
            // // stream << std::fixed << std::setprecision(2) ;
            // stream << std::setw(14)<< " " << "deflection angle from center_line: " << angle_center_v << " degree"<< std::endl;
            // stream << std::setw(14)<< " " << "deflection distance from center_line: " << sqrt(sdist_rotate_o) << " m"<< std::endl;
            // LOG(WARNING) << stream.str();




            // clean
            delete plane_v;
            delete plane_h;

        } else {
            LOG(WARNING) << "No truck!"; 
            delete hypothesis_;
            return ;
        }

        // clean
        delete hypothesis_;
                
        // see it
        std::vector<Graph::Vertex> g_edges, gg_edges;
        std::vector<Graph::Vertex> center_v_edges;

        auto models = viewer_->models();
        for (auto m : models) {
            // if (m->name() == "outcome" || m->name() == "center_v") {
            //     viewer_->deleteModel(m);
            // }

            if (m->name() == "outcome") {
                auto drawable_points = m->renderer()->get_points_drawable("vertices");
                auto drawable_lines  = m->renderer()->get_lines_drawable("edges");
                static std::vector<unsigned int> wire_indices = {0, 1, 1, 2, 2, 3, 3, 0, 4, 5,
                                                                 6, 7, 7, 8, 8, 9, 9, 6, 10, 11};
                std::vector<vec3> points;
                points.insert(points.end(), cgal_rect.begin(), cgal_rect.end());
                points.insert(points.end(), cgal_rect_project.begin(), cgal_rect_project.end());
                viewer_->makeCurrent();

                // void* pointer = VertexArrayObject::map_buffer(GL_ARRAY_BUFFER, drawable->vertex_buffer(), GL_WRITE_ONLY);
                // vec3* vertices = reinterpret_cast<vec3*>(pointer);
                // if (!vertices)
                //     return ;
                // VertexArrayObject::unmap_buffer(GL_ARRAY_BUFFER, drawable->vertex_buffer());


                drawable_points->update_vertex_buffer(points, true);
                drawable_lines->update_vertex_buffer(points, true);
                drawable_lines->update_element_buffer(wire_indices);

                viewer_->doneCurrent(); 
            }
            if (m->name() == "center_v") {
                auto drawable_points = m->renderer()->get_points_drawable("vertices");
                auto drawable_lines  = m->renderer()->get_lines_drawable("edges");
                static std::vector<unsigned int> wire_indices = {0, 1};
                std::vector<vec3> points;
                points.push_back(cgal_point_top);
                points.push_back(cgal_point_bottom);
                viewer_->makeCurrent();

                // void* pointer = VertexArrayObject::map_buffer(GL_ARRAY_BUFFER, drawable->vertex_buffer(), GL_WRITE_ONLY);
                // vec3* vertices = reinterpret_cast<vec3*>(pointer);
                // if (!vertices)
                //     return ;
                // VertexArrayObject::unmap_buffer(GL_ARRAY_BUFFER, drawable->vertex_buffer());


                drawable_points->update_vertex_buffer(points, true);
                drawable_lines->update_vertex_buffer(points, true);
                drawable_lines->update_element_buffer(wire_indices);

                viewer_->doneCurrent(); 
            }
            if (m->name() == "Plane_plus") {
                auto drawable_points = m->renderer()->get_points_drawable("vertices");
                auto drawable_lines  = m->renderer()->get_lines_drawable("edges");
                for (unsigned int i =0; i < alpha_points.size() -1; i = i +2) {
                    alpha_indices.push_back(i);
                    alpha_indices.push_back(i +1);
                }
                viewer_->makeCurrent();

                // void* pointer = VertexArrayObject::map_buffer(GL_ARRAY_BUFFER, drawable->vertex_buffer(), GL_WRITE_ONLY);
                // vec3* vertices = reinterpret_cast<vec3*>(pointer);
                // if (!vertices)
                //     return ;
                // VertexArrayObject::unmap_buffer(GL_ARRAY_BUFFER, drawable->vertex_buffer());


                drawable_points->update_vertex_buffer(alpha_points, true);
                drawable_lines->update_vertex_buffer(alpha_points, true);
                drawable_lines->update_element_buffer(alpha_indices);

                viewer_->doneCurrent(); 
            }
        }

        viewer_->update();

        return ;
/////////////////////////////////////////// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx //////////////////////////////////////////////////
        Graph* outcome = new Graph; 
        Graph* center_v = new Graph; 
        outcome->set_name("outcome");
        center_v->set_name("center_v");
#if 0
        for (std::size_t i = 0; i < interpoint.size(); i++) {
            g_edges.push_back(outcome->add_vertex(interpoint[i]));
        }
        g_edges.push_back(outcome->add_vertex(point_mid_top));
        g_edges.push_back(outcome->add_vertex(point_mid_bottom));
        for (std::size_t i = 0; i < rect.size(); i++) {
            gg_edges.push_back(outcome->add_vertex(rect[i]));
        }

        center_v_edges.push_back(center_v->add_vertex(point_top));
        center_v_edges.push_back(center_v->add_vertex(point_bottom));

        if (interpoint.size() == 4) {
            outcome->add_edge(g_edges[0], g_edges[1]);
            outcome->add_edge(g_edges[1], g_edges[2]);
            outcome->add_edge(g_edges[2], g_edges[3]);
            outcome->add_edge(g_edges[3], g_edges[0]);
        }
        outcome->add_edge(g_edges[4], g_edges[5]);
        center_v->add_edge(center_v_edges[0], center_v_edges[1]);
        if (rect.size() == 4) {
            outcome->add_edge(gg_edges[0], gg_edges[1]);
            outcome->add_edge(gg_edges[1], gg_edges[2]);
            outcome->add_edge(gg_edges[2], gg_edges[3]);
            outcome->add_edge(gg_edges[3], gg_edges[0]);
        }
#else
        center_v_edges.push_back(center_v->add_vertex(cgal_point_top));
        center_v_edges.push_back(center_v->add_vertex(cgal_point_bottom));
        center_v->add_edge(center_v_edges[0], center_v_edges[1]);

        for (std::size_t i = 0; i < cgal_rect.size(); i++) {
            g_edges.push_back(outcome->add_vertex(cgal_rect[i]));
            gg_edges.push_back(outcome->add_vertex(cgal_rect_project[i]));
        }
        outcome->add_edge(g_edges[0], g_edges[1]);
        outcome->add_edge(g_edges[1], g_edges[2]);
        outcome->add_edge(g_edges[2], g_edges[3]);
        outcome->add_edge(g_edges[3], g_edges[0]);
        outcome->add_edge(g_edges[4], g_edges[5]);

        outcome->add_edge(gg_edges[0], gg_edges[1]);
        outcome->add_edge(gg_edges[1], gg_edges[2]);
        outcome->add_edge(gg_edges[2], gg_edges[3]);
        outcome->add_edge(gg_edges[3], gg_edges[0]);
        outcome->add_edge(gg_edges[4], gg_edges[5]);

#endif

        viewer_->addModel(outcome);
        viewer_->addModel(center_v);

        // color it manual
        auto edges = outcome->renderer()->get_lines_drawable("edges");
        edges->set_uniform_coloring(vec4(0, 1, 0, 1));

        auto vertices = outcome->renderer()->get_points_drawable("vertices");
        vertices->set_uniform_coloring(vec4(1, 0, 0, 1));
        vertices->set_point_size(10.0f);

        // see it
        updateUi();
        viewer_->update();
    }
}

void MainWindow::readSettings()
{
    QSettings settings("liangliang.nan@gmail.com", "Mapple");
    recentFiles_ = settings.value("recentFiles").toStringList();
    updateRecentFileActions();
    curDataDirectory_ = settings.value("currentDirectory").toString();
}


void MainWindow::writeSettings()
{
    QSettings settings("liangliang.nan@gmail.com", "Mapple");
    settings.setValue("recentFiles", recentFiles_);
    if (!curDataDirectory_.isEmpty() && file_system::is_directory(curDataDirectory_.toStdString()))
        settings.setValue("currentDirectory", curDataDirectory_);
}


void MainWindow::updateWindowTitle() {
    Model* model = viewer_->currentModel();

#ifndef NDEBUG
    QString title = "Mapple (Debug Version)";
#else
    QString title = "Mapple";
#endif // _DEBUG

    QString fileName("Untitled");
    if (model)
        fileName = QString::fromStdString(model->name());

    title = tr("%1[*] - %2").arg(strippedName(fileName)).arg(title);
    setWindowTitle(title);
}


void MainWindow::closeEvent(QCloseEvent *event)
{
    if (okToContinue()) {
        writeSettings();
        event->accept();
    } else {
        event->ignore();
    }
}


void MainWindow::updateRecentFileActions()
{
    QMutableStringListIterator i(recentFiles_);
    while (i.hasNext()) {
        if (!QFile::exists(i.next()))
            i.remove();
    }

    for (int j = 0; j < MaxRecentFiles; ++j) {
        if (j < recentFiles_.count()) {
            QString text = tr("&%1 %2").arg(j + 1).arg(strippedName(recentFiles_[j]));
            actionsRecentFile[j]->setText(text);
            actionsRecentFile[j]->setData(recentFiles_[j]);
            actionsRecentFile[j]->setVisible(true);
        } else {
            actionsRecentFile[j]->setVisible(false);
        }
    }

    actionSeparator->setVisible(!recentFiles_.isEmpty());
}


QString MainWindow::strippedName(const QString &fullFileName)
{
    return QFileInfo(fullFileName).fileName();
}


void MainWindow::createActionsForFileMenu() {
    QActionGroup* actionGroup = new QActionGroup(this);
    actionGroup->addAction(ui->actionTranslateDisabled);
    actionGroup->addAction(ui->actionTranslateUseFirstVertex);
    actionGroup->addAction(ui->actionTranslateUseLastKnownVertex);
    connect(actionGroup, SIGNAL(triggered(QAction*)), this, SLOT(loadModelTranslateChanged(QAction*)));

    connect(ui->actionOpen, SIGNAL(triggered()), this, SLOT(onOpen()));
    connect(ui->actionSave, SIGNAL(triggered()), this, SLOT(onSave()));
    actionSeparator = ui->menuFile->addSeparator();

    QList<QAction*> actions;
    for (int i = 0; i < MaxRecentFiles; ++i) {
        actionsRecentFile[i] = new QAction(this);
        actionsRecentFile[i]->setVisible(false);
        connect(actionsRecentFile[i], SIGNAL(triggered()), this, SLOT(onOpenRecentFile()));

        actions.push_back(actionsRecentFile[i]);
    }
    ui->menuRecentFiles->insertActions(ui->actionClearRecentFiles, actions);
    ui->menuRecentFiles->insertSeparator(ui->actionClearRecentFiles);
    connect(ui->actionClearRecentFiles, SIGNAL(triggered()), this, SLOT(onClearRecentFiles()));

    connect(ui->actionExit, SIGNAL(triggered()), this, SLOT(close()));
    ui->actionExit->setShortcut(QString("Ctrl+Q"));
}


void MainWindow::createActionsForViewMenu() {
    connect(ui->actionShowPrimitiveIDUnderMouse, SIGNAL(toggled(bool)), viewer_, SLOT(showPrimitiveIDUnderMouse(bool)));
    connect(ui->actionShowPrimitivePropertyUnderMouse, SIGNAL(toggled(bool)), this, SLOT(showPrimitivePropertyUnderMouse(bool)));
    connect(ui->actionShowCordinatesUnderMouse, SIGNAL(toggled(bool)), this, SLOT(showCoordinatesUnderMouse(bool)));

    connect(ui->actionShowEasy3DLogo, SIGNAL(toggled(bool)), viewer_, SLOT(showEasy3DLogo(bool)));
    connect(ui->actionShowFrameRate, SIGNAL(toggled(bool)), viewer_, SLOT(showFrameRate(bool)));
    connect(ui->actionShowAxes, SIGNAL(toggled(bool)), viewer_, SLOT(showAxes(bool)));

    connect(ui->actionShowCameraPath, SIGNAL(toggled(bool)), this, SLOT(setShowCameraPath(bool)));
    connect(ui->actionShowKeyframeCameras, SIGNAL(toggled(bool)), this, SLOT(setShowKeyframeCameras(bool)));

    QAction* actionToggleDockWidgetRendering = ui->dockWidgetRendering->toggleViewAction();
    actionToggleDockWidgetRendering->setText("Rendering Panel");
    ui->menuView->addAction(actionToggleDockWidgetRendering);

    QAction* actionToggleDockWidgetModels = ui->dockWidgetModels->toggleViewAction();
    actionToggleDockWidgetModels->setText("Model Panel");
    ui->menuView->addAction(actionToggleDockWidgetModels);

    QAction* actionToggleDockWidgetLogger = ui->dockWidgetLog->toggleViewAction();
    actionToggleDockWidgetLogger->setText("Log Panel");
    ui->menuView->addAction(actionToggleDockWidgetLogger);

    connect(ui->actionSetBackgroundColor, SIGNAL(triggered()), this, SLOT(setBackgroundColor()));
}


void MainWindow::createActionsForCameraMenu() {
    connect(ui->actionPerspectiveOrthographic, SIGNAL(toggled(bool)), viewer_, SLOT(setPerspective(bool)));
    connect(ui->actionFitScreen, SIGNAL(triggered()), viewer_, SLOT(fitScreen()));
    connect(ui->actionSnapshot, SIGNAL(triggered()), this, SLOT(saveSnapshot()));

    connect(ui->actionCopyCamera, SIGNAL(triggered()), viewer_, SLOT(copyCamera()));
    connect(ui->actionPasteCamera, SIGNAL(triggered()), viewer_, SLOT(pasteCamera()));

    connect(ui->actionSaveCameraStateToFile, SIGNAL(triggered()), this, SLOT(saveCameraStateToFile()));
    connect(ui->actionRestoreCameraStateFromFile, SIGNAL(triggered()), this, SLOT(restoreCameraStateFromFile()));

    connect(ui->actionImportCameraPath, SIGNAL(triggered()), this, SLOT(importCameraPath()));
    connect(ui->actionExportCameraPath, SIGNAL(triggered()), this, SLOT(exportCameraPath()));
    connect(ui->actionAnimation, SIGNAL(triggered()), this, SLOT(animation()));
}


void MainWindow::createActionsForPropertyMenu() {
    connect(ui->actionManipulateProperties, SIGNAL(triggered()), this, SLOT(manipulateProperties()));
    connect(ui->actionComputeHeightField, SIGNAL(triggered()), this, SLOT(computeHeightField()));
    connect(ui->actionComputeSurfaceMeshCurvatures, SIGNAL(triggered()), this, SLOT(computeSurfaceMeshCurvatures()));
    connect(ui->actionTopologyStatistics, SIGNAL(triggered()), this, SLOT(reportTopologyStatistics()));
}


void MainWindow::createActionsForEditMenu() {
    connect(ui->actionTranslationalRecenter, SIGNAL(triggered()), this, SLOT(translationalRecenter()));
    connect(ui->actionAddGaussianNoise, SIGNAL(triggered()), this, SLOT(addGaussianNoise()));
    connect(ui->actionApplyManipulatedTransformation, SIGNAL(triggered()), this, SLOT(applyManipulatedTransformation()));
    connect(ui->actionGiveUpManipulatedTransformation, SIGNAL(triggered()), this, SLOT(giveUpManipulatedTransformation()));

    connect(ui->actionGenerateColorPropertyFromIndexedColors, SIGNAL(triggered()), this, SLOT(generateColorPropertyFromIndexedColors()));
}


void MainWindow::createActionsForSelectMenu() {
    connect(ui->actionSelectModel, SIGNAL(toggled(bool)), viewer_, SLOT(enableSelectModel(bool)));

    connect(ui->actionInvertSelection, SIGNAL(triggered()), viewer_, SLOT(invertSelection()));
    connect(ui->actionDeleteSelectedPrimitives, SIGNAL(triggered()), viewer_, SLOT(deleteSelectedPrimitives()));

    //////////////////////////////////////////////////////////////////////////

    QActionGroup* actionGroup = new QActionGroup(this);
    actionGroup->addAction(ui->actionCameraManipulation);
    actionGroup->addAction(ui->actionSelectClick);
    actionGroup->addAction(ui->actionSelectRect);
    actionGroup->addAction(ui->actionSelectLasso);

    connect(actionGroup, SIGNAL(triggered(QAction*)), this, SLOT(operationModeChanged(QAction*)));
}


void MainWindow::createActionsForPointCloudMenu() {
    connect(ui->actionDownSampling, SIGNAL(triggered()), this, SLOT(pointCloudDownsampling()));

    connect(ui->actionEstimatePointCloudNormals, SIGNAL(triggered()), this, SLOT(pointCloudEstimateNormals()));
    connect(ui->actionReorientPointCloudNormals, SIGNAL(triggered()), this, SLOT(pointCloudReorientNormals()));
    connect(ui->actionNormalizePointCloudNormals, SIGNAL(triggered()), this, SLOT(pointCloudNormalizeNormals()));

    connect(ui->actionRansacPrimitiveExtraction, SIGNAL(triggered()), this, SLOT(pointCloudRansacPrimitiveExtraction()));
    connect(ui->actionPoissonSurfaceReconstruction, SIGNAL(triggered()), this, SLOT(pointCloudPoissonSurfaceReconstruction()));

    connect(ui->actionDelaunayTriangulation2D, SIGNAL(triggered()), this, SLOT(pointCloudDelaunayTriangulation2D()));
    connect(ui->actionDelaunayTriangulation3D, SIGNAL(triggered()), this, SLOT(pointCloudDelaunayTriangulation3D()));
}


void MainWindow::createActionsForSurfaceMeshMenu() {
    connect(ui->actionExtractConnectedComponents, SIGNAL(triggered()), this, SLOT(surfaceMeshExtractConnectedComponents()));
    connect(ui->actionDualMesh, SIGNAL(triggered()), this, SLOT(surfaceMeshDual()));
    connect(ui->actionPlanarPartition, SIGNAL(triggered()), this, SLOT(surfaceMeshPlanarPartition()));
    connect(ui->actionPolygonization, SIGNAL(triggered()), this, SLOT(surfaceMeshPolygonization()));
    connect(ui->actionSurfaceMeshTriangulation, SIGNAL(triggered()), this, SLOT(surfaceMeshTriangulation()));
    connect(ui->actionSurfaceMeshTetrahedralization, SIGNAL(triggered()), this, SLOT(surfaceMeshTetrahedralization()));

    connect(ui->actionSurfaceMeshRepairPolygonSoup, SIGNAL(triggered()), this, SLOT(surfaceMeshRepairPolygonSoup()));
    connect(ui->actionSurfaceMeshOrientAndStitchPolygonSoup, SIGNAL(triggered()), this, SLOT(surfaceMeshOrientAndStitchPolygonSoup()));

    connect(ui->actionSurfaceMeshClip, SIGNAL(triggered()), this, SLOT(surfaceMeshClip()));
    connect(ui->actionSurfaceMeshSplit, SIGNAL(triggered()), this, SLOT(surfaceMeshSplit()));
    connect(ui->actionSurfaceMeshSlice, SIGNAL(triggered()), this, SLOT(surfaceMeshSlice()));

    connect(ui->actionStitchWithReorientation, SIGNAL(triggered()), this, SLOT(surfaceMeshStitchWithReorientation()));
    connect(ui->actionStitchWithoutReorientation, SIGNAL(triggered()), this, SLOT(surfaceMeshStitchWithoutReorientation()));

    connect(ui->actionOrientClosedTriangleMesh, SIGNAL(triggered()), this, SLOT(surfaceMeshOrientClosedTriangleMesh()));
    connect(ui->actionReverseOrientation, SIGNAL(triggered()), this, SLOT(surfaceMeshReverseOrientation()));

    connect(ui->actionSurfaceMeshRemoveIsolatedVertices, SIGNAL(triggered()), this, SLOT(surfaceMeshRemoveIsolatedVertices()));
    connect(ui->actionRemoveDuplicateAndFoldingFaces, SIGNAL(triggered()), this, SLOT(surfaceMeshRemoveDuplicateAndFoldingFaces()));
    connect(ui->actionDetectSelfIntersections, SIGNAL(triggered()), this, SLOT(surfaceMeshDetectSelfIntersections()));
    connect(ui->actionRemeshSelfIntersections, SIGNAL(triggered()), this, SLOT(surfaceMeshRemeshSelfIntersections()));

    connect(ui->actionSurfaceMeshSubdivisionCatmullClark, SIGNAL(triggered()), this, SLOT(surfaceMeshSubdivisionCatmullClark()));
    connect(ui->actionSurfaceMeshSubdivisionLoop, SIGNAL(triggered()), this, SLOT(surfaceMeshSubdivisionLoop()));
    connect(ui->actionSurfaceMeshSubdivisionSqrt3, SIGNAL(triggered()), this, SLOT(surfaceMeshSubdivisionSqrt3()));

    connect(ui->actionSurfaceMeshFairing, SIGNAL(triggered()), this, SLOT(surfaceMeshFairing()));
    connect(ui->actionSurfaceMeshSmoothing, SIGNAL(triggered()), this, SLOT(surfaceMeshSmoothing()));
    connect(ui->actionSurfaceMeshHoleFilling, SIGNAL(triggered()), this, SLOT(surfaceMeshHoleFilling()));
    connect(ui->actionSurfaceMeshSimplification, SIGNAL(triggered()), this, SLOT(surfaceMeshSimplification()));
    connect(ui->actionSurfaceMeshParameterization, SIGNAL(triggered()), this, SLOT(surfaceMeshParameterization()));
    connect(ui->actionSurfaceMeshRemeshing, SIGNAL(triggered()), this, SLOT(surfaceMeshRemeshing()));
    connect(ui->actionSurfaceMeshGeodesic, SIGNAL(triggered()), this, SLOT(surfaceMeshGeodesic()));
    connect(ui->actionSamplingSurfaceMesh, SIGNAL(triggered()), this, SLOT(surfaceMeshSampling()));
    connect(ui->actionCreateSurfaceMeshFromText, SIGNAL(triggered()), this, SLOT(surfaceMeshCreateMeshFromText()));
}


void MainWindow::createActionsForPolyMeshMenu() {
    connect(ui->actionPolyMeshExtractBoundary, SIGNAL(triggered()), this, SLOT(polymeshExtractBoundary()));
}


void MainWindow::operationModeChanged(QAction* act) {
    if (act == ui->actionCameraManipulation) {
        viewer()->tool_manager()->set_tool(tools::ToolManager::EMPTY_TOOL);
        return;
    }

    auto tool_manager = viewer()->tool_manager();
    if (act == ui->actionSelectClick) {
        if (dynamic_cast<SurfaceMesh *>(viewer()->currentModel()))
            tool_manager->set_tool(tools::ToolManager::SELECT_SURFACE_MESH_FACE_CLICK_TOOL);
        else if (dynamic_cast<PointCloud *>(viewer()->currentModel()))
            tool_manager->set_tool(tools::ToolManager::SELECT_POINT_CLOUD_CLICK_TOOL);
    }
    else if (act == ui->actionSelectRect) {
        if (dynamic_cast<SurfaceMesh *>(viewer()->currentModel()))
            tool_manager->set_tool(tools::ToolManager::SELECT_SURFACE_MESH_FACE_RECT_TOOL);
        else if (dynamic_cast<PointCloud *>(viewer()->currentModel()))
            tool_manager->set_tool(tools::ToolManager::SELECT_POINT_CLOUD_RECT_TOOL);
    }
    else if (act == ui->actionSelectLasso) {
        if (dynamic_cast<SurfaceMesh *>(viewer()->currentModel()))
            tool_manager->set_tool(tools::ToolManager::SELECT_SURFACE_MESH_FACE_LASSO_TOOL);
        else if (dynamic_cast<PointCloud *>(viewer()->currentModel()))
            tool_manager->set_tool(tools::ToolManager::SELECT_POINT_CLOUD_LASSO_TOOL);
    }

    if (viewer()->tool_manager()->current_tool())
        statusBar()->showMessage(QString::fromStdString(tool_manager->current_tool()->instruction()), 2000);
}


void MainWindow::reportTopologyStatistics() {
    SurfaceMesh *mesh = dynamic_cast<SurfaceMesh *>(viewer()->currentModel());
    if (mesh) {
        std::stringstream stream;

        const auto &components = SurfaceMeshComponent::extract(mesh);
        stream << "model has " << components.size() << " connected components";

        // count isolated vertices
        std::size_t count = 0;
        for (auto v : mesh->vertices()) {
            if (mesh->is_isolated(v))
                ++count;
        }
        if (count > 0)
            stream << "and " << count << " isolated vertices";
        stream << std::endl;

        const std::size_t num = 10;
        if (components.size() > num)
            stream << "    topology of the first " << num << " components:" << std::endl;

        for (std::size_t i = 0; i < std::min(components.size(), num); ++i) {
            const SurfaceMeshComponent &comp = components[i];
            SurfaceMeshTopology topo(&comp);
            std::string type = "unknown";
            if (topo.is_sphere())
                type = "sphere";
            else if (topo.is_disc())
                type = "disc";
            else if (topo.is_cylinder())
                type = "cylinder";
            else if (topo.is_torus())
                type = "torus";
            else if (topo.is_closed())
                type = "unknown closed";

            stream << "        " << i << ": " << type
                   << ", F = " << comp.n_faces() << ", V = " << comp.n_vertices() << ", E = " << comp.n_edges()
                   << ", B = " << topo.number_of_borders();
            if (topo.number_of_borders() == 1)
                stream << ", border size = " << topo.largest_border_size();
            else if (topo.number_of_borders() > 1)
                stream << ", largest border size = " << topo.largest_border_size();
            stream << std::endl;
        }

        LOG(INFO) << stream.str();
    }

    Graph *graph = dynamic_cast<Graph *>(viewer()->currentModel());
    if (graph) {
        std::stringstream stream;
        stream << "graph has " << graph->n_vertices() << " vertices and " << graph->n_edges() << " edges" << std::endl;

        std::map<int, int> count; // key (i.e., first element) is the valence
        for (auto v : graph->vertices())
            ++count[graph->valence(v)];

        for (const auto& elem : count)
            stream << "    number of degree " << elem.first << " vertices: " << elem.second << std::endl;

        LOG(INFO) << stream.str();
    }
}


void MainWindow::surfaceMeshTriangulation() {
    auto mesh = dynamic_cast<SurfaceMesh*>(viewer_->currentModel());
    if (!mesh)
        return;

    SurfaceMeshTriangulation triangulator(mesh);
    triangulator.triangulate(SurfaceMeshTriangulation::MIN_AREA);

    mesh->renderer()->update();
    viewer_->update();
    updateUi();
}


void MainWindow::surfaceMeshTetrahedralization() {
    auto mesh = dynamic_cast<SurfaceMesh*>(viewer_->currentModel());
    if (!mesh)
        return;

    SurfaceMeshTetrehedralization tetra;
    PolyMesh* result = tetra.apply(mesh);
    if (result) {
        const std::string &name = file_system::name_less_extension(mesh->name()) + "_tetrahedralization.plm";
        result->set_name(name);

        viewer_->addModel(result);
        updateUi();
        viewer_->update();
    }
}


void MainWindow::surfaceMeshRepairPolygonSoup() {
    SurfaceMesh* mesh = dynamic_cast<SurfaceMesh*>(viewer()->currentModel());
    if (!mesh)
        return;

#if HAS_CGAL
    Surfacer::repair_polygon_soup(mesh);

    mesh->renderer()->update();
    viewer_->update();
    updateUi();
#else
    LOG(WARNING) << "This function requires CGAL but CGAL was not found when Easy3D was built.";
#endif
}


void MainWindow::surfaceMeshStitchWithReorientation() {
    SurfaceMesh* mesh = dynamic_cast<SurfaceMesh*>(viewer()->currentModel());
    if (!mesh)
        return;

#if HAS_CGAL
    Surfacer::stitch_borders(mesh);
    Surfacer::merge_reversible_connected_components(mesh);
#else
    SurfaceMeshStitching stitch(mesh);
    stitch.apply();
    LOG(WARNING) << "install CGAL to allow stitching connected components with incompatible boundaries";
#endif

    mesh->renderer()->update();
    viewer_->update();
    updateUi();
}


void MainWindow::surfaceMeshOrientAndStitchPolygonSoup() {
    SurfaceMesh* mesh = dynamic_cast<SurfaceMesh*>(viewer()->currentModel());
    if (!mesh)
        return;

#if HAS_CGAL
    Surfacer::orient_and_stitch_polygon_soup(mesh);
#else
    SurfaceMeshStitching stitch(mesh);
    stitch.apply();
    LOG(WARNING) << "install CGAL to allow stitching connected components with incompatible boundaries";
#endif

    mesh->renderer()->update();
    viewer_->update();
    updateUi();
}


void MainWindow::surfaceMeshStitchWithoutReorientation() {
    SurfaceMesh* mesh = dynamic_cast<SurfaceMesh*>(viewer()->currentModel());
    if (!mesh)
        return;

#if HAS_CGAL
    Surfacer::stitch_borders(mesh);
#else
    SurfaceMeshStitching stitch(mesh);
    stitch.apply();
    LOG(WARNING) << "install CGAL to allow stitching connected components with incompatible boundaries";
#endif

    mesh->renderer()->update();
    viewer_->update();
    updateUi();
}


void MainWindow::surfaceMeshOrientClosedTriangleMesh() {
    SurfaceMesh* mesh = dynamic_cast<SurfaceMesh*>(viewer()->currentModel());
    if (!mesh)
        return;

#if HAS_CGAL
    Surfacer::orient_closed_triangle_mesh(mesh);

    mesh->renderer()->update();
    viewer_->update();
    updateUi();
#else
    LOG(WARNING) << "This function requires CGAL but CGAL was not found when Easy3D was built.";
#endif
}


void MainWindow::surfaceMeshReverseOrientation() {
    SurfaceMesh* mesh = dynamic_cast<SurfaceMesh*>(viewer()->currentModel());
    if (!mesh)
        return;

    mesh->reverse_orientation();

    mesh->renderer()->update();
    viewer_->update();
    updateUi();
}


void MainWindow::surfaceMeshRemoveIsolatedVertices() {
    SurfaceMesh *mesh = dynamic_cast<SurfaceMesh *>(viewer()->currentModel());
    if (!mesh)
        return;

    unsigned int prev_num_vertices = mesh->n_vertices();

    // clean: remove isolated vertices
    for (auto v : mesh->vertices()) {
        if (mesh->is_isolated(v))
            mesh->delete_vertex(v);
    }
    mesh->collect_garbage();

    unsigned int count = prev_num_vertices - mesh->n_vertices();
    LOG(INFO) << count << " isolated vertices deleted.";

    mesh->renderer()->update();
    viewer_->update();
    updateUi();
}


void MainWindow::surfaceMeshRemoveDuplicateAndFoldingFaces() {
    SurfaceMesh* mesh = dynamic_cast<SurfaceMesh*>(viewer()->currentModel());
    if (!mesh)
        return;

#if HAS_CGAL
    StopWatch w;
    w.start();
	LOG(INFO) << "removing overlapping faces...";

    unsigned int num_degenerate = Surfacer::remove_degenerate_faces(mesh, 1e-5);
    unsigned int num_overlapping = Surfacer::remove_overlapping_faces(mesh, true);
    if (num_degenerate + num_overlapping > 0) {
        mesh->renderer()->update();
        viewer_->update();
        updateUi();
    }
    LOG(INFO) << "done. " << num_degenerate + num_overlapping << " faces removed (" << num_degenerate
              << " degenerate, " << num_overlapping << " overlapping). " << w.time_string();
#else
    LOG(WARNING) << "This function requires CGAL but CGAL was not found when Easy3D was built.";
#endif
}


void MainWindow::surfaceMeshDetectSelfIntersections() {
    SurfaceMesh* mesh = dynamic_cast<SurfaceMesh*>(viewer()->currentModel());
    if (!mesh)
        return;

#if HAS_CGAL
    StopWatch w;
    w.start();
	LOG(INFO) << "detecting intersecting faces...";

    const auto& pairs = Surfacer::detect_self_intersections(mesh);
    if (pairs.empty())
        LOG(INFO) << "done. No intersecting faces detected. " << w.time_string();
    else {
        auto select = mesh->get_face_property<bool>("f:select");
        if (select)
            select.vector().resize(mesh->n_faces(), false);
        else
            select = mesh->add_face_property<bool>("f:select", false);

        for (const auto& pair : pairs) {
            select[pair.first] = true;
            select[pair.second] = true;
        }

        LOG(INFO) << "done. " << pairs.size() << " pairs of faces intersect (marked in face property 'f:select'). " << w.time_string();
        updateRenderingPanel();
    }
#else
    LOG(WARNING) << "This function requires CGAL but CGAL was not found when Easy3D was built.";
#endif
 }


void MainWindow::surfaceMeshRemeshSelfIntersections() {
    SurfaceMesh* mesh = dynamic_cast<SurfaceMesh*>(viewer()->currentModel());
    if (!mesh)
        return;

#if HAS_CGAL
    StopWatch w;
    w.start();
	LOG(INFO) << "remeshing self intersections...";

    auto size = mesh->n_faces();
    if (Surfacer::remesh_self_intersections(mesh, true)) {
		LOG(INFO) << "done. #faces " << size << " -> " << mesh->n_faces() << ". " << w.time_string();
        mesh->renderer()->update();
        viewer_->update();
        updateUi();
    }
    else
		LOG(INFO) << "done. No intersecting faces detected. " << w.time_string();
#else
    LOG(WARNING) << "This function requires CGAL but CGAL was not found when Easy3D was built.";
#endif
}


void MainWindow::surfaceMeshClip() {
    SurfaceMesh* mesh = dynamic_cast<SurfaceMesh*>(viewer()->currentModel());
    if (!mesh)
        return;

#if HAS_CGAL
    auto clipping_plane = easy3d::setting::clipping_plane;
    if (!clipping_plane || !clipping_plane->is_enabled()) {
        LOG(WARNING) << "clipping plane is not defined";
        return;
    }

    Surfacer::clip(mesh, clipping_plane->plane0(), false);

    mesh->renderer()->update();
    viewer_->update();
    updateUi();
#else
    LOG(WARNING) << "This function requires CGAL but CGAL was not found when Easy3D was built.";
#endif
}


void MainWindow::surfaceMeshSplit() {
    SurfaceMesh* mesh = dynamic_cast<SurfaceMesh*>(viewer()->currentModel());
    if (!mesh)
        return;

#if HAS_CGAL
    auto clipping_plane = easy3d::setting::clipping_plane;
    if (!clipping_plane || !clipping_plane->is_enabled()) {
        LOG(WARNING) << "clipping plane is not defined";
        return;
    }

    Surfacer::split(mesh, clipping_plane->plane0());

    mesh->renderer()->update();
    viewer_->update();
    updateUi();
#else
    LOG(WARNING) << "This function requires CGAL but CGAL was not found when Easy3D was built.";
#endif
}


void MainWindow::surfaceMeshSlice() {
    SurfaceMesh* mesh = dynamic_cast<SurfaceMesh*>(viewer()->currentModel());
    if (!mesh)
        return;

#if HAS_CGAL

#if 0 // slice by the visual clipping plane

    auto clipping_plane = easy3d::setting::clipping_plane;
    if (!clipping_plane || !clipping_plane->is_enabled()) {
        LOG(WARNING) << "clipping plane is not defined";
        return;
    }

    const std::vector<Surfacer::Polyline>& polylines = Surfacer::slice(mesh, clipping_plane->plane0());

    Graph* graph = new Graph;
    for (const auto& polyline : polylines) {
        for (const auto &p : polyline)
            graph->add_vertex(p);
    }

    unsigned int idx = 0;
    for (const auto& polyline : polylines) {
        for (unsigned int i=0; i<polyline.size() - 1; ++i) {
            graph->add_edge(Graph::Vertex(idx), Graph::Vertex(idx + 1));
            ++idx;
        }
        ++idx;
    }

    graph->set_name(file_system::base_name(mesh->name()) + "-slice");
    viewer()->addModel(graph);
    ui->treeWidgetModels->addModel(graph, false);

#else   // slices using a set of horizontal planes

    float minz = mesh->bounding_box().min_point().z;
    float maxz = mesh->bounding_box().max_point().z;

    int num = 10;
    float step = (maxz - minz) / num;

    std::vector<Plane3> planes(num);
    for (int i=0; i<num; ++i)
        planes[i] = Plane3(vec3(0, 0, minz + i * step), vec3(0, 0, 1));

    const std::vector< std::vector<Surfacer::Polyline> >& all_polylines = Surfacer::slice(mesh, planes);
    if (all_polylines.empty())
        return;

    Graph* graph = new Graph;
    for (const auto& polylines : all_polylines) {
        for (const auto &polyline : polylines) {
            for (const auto &p : polyline)
                graph->add_vertex(p);
        }
    }

    auto color = graph->add_edge_property<vec3>("e:color");
    unsigned int idx = 0;
    for (const auto& polylines : all_polylines) {
        for (const auto &polyline : polylines) {
            const auto& c = random_color();
            for (unsigned int i = 0; i < polyline.size() - 1; ++i) {
                auto e = graph->add_edge(Graph::Vertex(idx), Graph::Vertex(idx + 1));
                color[e] = c;
                ++idx;
            }
            ++idx;
        }
    }

    graph->set_name(file_system::base_name(mesh->name()) + "-slice");
    viewer()->addModel(graph);

    auto edges = graph->renderer()->get_lines_drawable("edges");
    edges->set_line_width(2.0f);
    edges->set_uniform_coloring(vec4(1, 0, 0, 1));
    LOG(INFO) << "color information added to visualize individual polylines of the slice";
    edges->set_coloring(easy3d::State::COLOR_PROPERTY, easy3d::State::EDGE, "e:color");

    auto vertices = graph->renderer()->get_points_drawable("vertices");
    vertices->set_uniform_coloring(vec4(0, 1, 0, 1));
    vertices->set_point_size(4.0f);
    vertices->set_visible(false);

    ui->treeWidgetModels->addModel(graph, false);
#endif

#else
    LOG(WARNING) << "This function requires CGAL but CGAL was not found when Easy3D was built.";
#endif
}


void MainWindow::surfaceMeshCreateMeshFromText() {
    static DialogSurfaceMeshFromText* dialog = nullptr;
    if (!dialog)
        dialog = new DialogSurfaceMeshFromText(this);
    dialog->show();
}


void MainWindow::pointCloudEstimateNormals() {
    PointCloud* cloud = dynamic_cast<PointCloud*>(viewer()->currentModel());
    if (!cloud)
        return;

    DialogPointCloudNormalEstimation dlg(this);
    if (dlg.exec() == QDialog::Accepted) {
        unsigned int k = dlg.lineEditNeighborSize->text().toUInt();
        PointCloudNormals pcn;
        pcn.estimate(cloud, k);
        cloud->renderer()->update();
        viewer()->update();
    }
}


void MainWindow::pointCloudReorientNormals() {
    PointCloud *cloud = dynamic_cast<PointCloud *>(viewer()->currentModel());
    if (!cloud)
        return;

    DialogPointCloudNormalEstimation dlg(this);
    if (dlg.exec() == QDialog::Accepted) {
        unsigned int k = dlg.lineEditNeighborSize->text().toUInt();
        PointCloudNormals pcn;
        pcn.reorient(cloud, k);
        cloud->renderer()->update();
        viewer()->update();
    }
}


void MainWindow::pointCloudNormalizeNormals() {
    PointCloud* cloud = dynamic_cast<PointCloud*>(viewer()->currentModel());
    if (!cloud)
        return;

    auto prop = cloud->get_vertex_property<vec3>("v:normal");
    if (!prop) {
        LOG(WARNING) << "point cloud does not have normal information";
        return;
    }

    auto &normals = prop.vector();
    for (auto &n : normals)
        n.normalize();

    cloud->renderer()->update();
    viewer()->update();
}


void MainWindow::polymeshExtractBoundary() {
    PolyMesh* poly = dynamic_cast<PolyMesh*>(viewer()->currentModel());
    if (!poly)
        return;

    std::vector<std::vector<PolyMesh::Vertex> > faces;
    poly->extract_boundary(faces);

    std::unordered_map<PolyMesh::Vertex, SurfaceMesh::Vertex, PolyMesh::Vertex::Hash> unique_vertex;


    SurfaceMesh* mesh = new SurfaceMesh;
    const std::string &name = file_system::name_less_extension(poly->name()) + "_boundary.ply";
    mesh->set_name(name);

    SurfaceMeshBuilder builder(mesh);
    builder.begin_surface();
    for (auto f : faces) {
        std::vector<SurfaceMesh::Vertex> vts;
        for (auto pv : f) {
            auto pos = unique_vertex.find(pv);
            if (pos == unique_vertex.end()) {
                auto sv = builder.add_vertex(poly->position(pv));
                unique_vertex[pv] = sv;
                vts.push_back(sv);
            } else
                vts.push_back(pos->second);
        }
        builder.add_face(vts);
    }
    builder.end_surface();

    viewer_->addModel(mesh);
    updateUi();
    viewer_->update();
}


void MainWindow::computeHeightField() {
    auto model = viewer_->currentModel();
    if (!model)
        return;

    // add 3 scalar fields defined on vertices, edges, and faces respectively.
    if (dynamic_cast<SurfaceMesh*>(model)) {
        SurfaceMesh* mesh = dynamic_cast<SurfaceMesh*>(model);

        ProgressLogger progress(4, false, false);

        auto v_height_x = mesh->vertex_property<float>("v:height_x");
        auto v_height_y = mesh->vertex_property<float>("v:height_y");
        auto v_height_z = mesh->vertex_property<float>("v:height_z");
        for (auto v : mesh->vertices()) {
            const auto& p = mesh->position(v);
            v_height_x[v] = p.x;
            v_height_y[v] = p.y;
            v_height_z[v] = p.z;
        }
        progress.next();

        auto e_height_x = mesh->edge_property<float>("e:height_x");
        auto e_height_y = mesh->edge_property<float>("e:height_y");
        auto e_height_z = mesh->edge_property<float>("e:height_z");
        for (auto e : mesh->edges()) {
            const auto& s = mesh->vertex(e, 0);
            const auto& t = mesh->vertex(e, 1);
            const auto& c = 0.5 * (mesh->position(s) + mesh->position(t));
            e_height_x[e] = c.x;
            e_height_y[e] = c.y;
            e_height_z[e] = c.z;
        }
        progress.next();

        auto f_height_x = mesh->face_property<float>("f:height_x");
        auto f_height_y = mesh->face_property<float>("f:height_y");
        auto f_height_z = mesh->face_property<float>("f:height_z");
        for (auto f : mesh->faces()) {
            vec3 c(0,0,0);
            float count = 0.0f;
            for (auto v : mesh->vertices(f)) {
                c += mesh->position(v);
                ++count;
            }
            c /= count;
            f_height_x[f] = c.x;
            f_height_y[f] = c.y;
            f_height_z[f] = c.z;
        }
        progress.next();

        // add a vector field to the faces
        mesh->update_face_normals();
        auto fnormals = mesh->get_face_property<vec3>("f:normal");

        // add a vector field to the edges
        auto enormals = mesh->edge_property<vec3>("e:normal");
        for (auto e : mesh->edges()) {
            vec3 n(0,0,0);
            float count(0.0f);
            auto f = mesh->face(e, 0);
            if (f.is_valid()) {
                n += fnormals[f];
                count += 1.0f;
            }
            f = mesh->face(e, 1);
            if (f.is_valid()) {
                n += fnormals[f];
                count += 1.0f;
            }
            enormals[e] = n.normalize();
        }
        progress.next();
    }

    else if (dynamic_cast<PointCloud*>(model)) {
        PointCloud* cloud = dynamic_cast<PointCloud*>(model);

        auto v_height_x = cloud->vertex_property<float>("v:height_x");
        auto v_height_y = cloud->vertex_property<float>("v:height_y");
        auto v_height_z = cloud->vertex_property<float>("v:height_z");
        for (auto v : cloud->vertices()) {
            const auto& p = cloud->position(v);
            v_height_x[v] = p.x;
            v_height_y[v] = p.y;
            v_height_z[v] = p.z;
        }
    }

    else if (dynamic_cast<Graph*>(model)) {
        Graph* graph = dynamic_cast<Graph*>(model);

        auto v_height_x = graph->vertex_property<float>("v:height_x");
        auto v_height_y = graph->vertex_property<float>("v:height_y");
        auto v_height_z = graph->vertex_property<float>("v:height_z");
        for (auto v : graph->vertices()) {
            const auto& p = graph->position(v);
            v_height_x[v] = p.x;
            v_height_y[v] = p.y;
            v_height_z[v] = p.z;
        }

        auto e_height_x = graph->edge_property<float>("e:height_x");
        auto e_height_y = graph->edge_property<float>("e:height_y");
        auto e_height_z = graph->edge_property<float>("e:height_z");
        for (auto e : graph->edges()) {
            const auto& s = graph->vertex(e, 0);
            const auto& t = graph->vertex(e, 1);
            const auto& c = 0.5 * (graph->position(s) + graph->position(t));
            e_height_x[e] = c.x;
            e_height_y[e] = c.y;
            e_height_z[e] = c.z;
        }
    }
    // add 3 scalar fields defined on vertices, edges, and faces respectively.
    else if (dynamic_cast<PolyMesh*>(model)) {
        PolyMesh* mesh = dynamic_cast<PolyMesh*>(model);

        ProgressLogger progress(4, false, false);

        auto v_height_x = mesh->vertex_property<float>("v:height_x");
        auto v_height_y = mesh->vertex_property<float>("v:height_y");
        auto v_height_z = mesh->vertex_property<float>("v:height_z");
        for (auto v : mesh->vertices()) {
            const auto& p = mesh->position(v);
            v_height_x[v] = p.x;
            v_height_y[v] = p.y;
            v_height_z[v] = p.z;
        }
        progress.next();

        auto e_height_x = mesh->edge_property<float>("e:height_x");
        auto e_height_y = mesh->edge_property<float>("e:height_y");
        auto e_height_z = mesh->edge_property<float>("e:height_z");
        for (auto e : mesh->edges()) {
            const auto& s = mesh->vertex(e, 0);
            const auto& t = mesh->vertex(e, 1);
            const auto& c = 0.5 * (mesh->position(s) + mesh->position(t));
            e_height_x[e] = c.x;
            e_height_y[e] = c.y;
            e_height_z[e] = c.z;
        }
        progress.next();

        auto f_height_x = mesh->face_property<float>("f:height_x");
        auto f_height_y = mesh->face_property<float>("f:height_y");
        auto f_height_z = mesh->face_property<float>("f:height_z");
        for (auto f : mesh->faces()) {
            vec3 c(0,0,0);
            float count = 0.0f;
            for (auto v : mesh->vertices(f)) {
                c += mesh->position(v);
                ++count;
            }
            c /= count;
            f_height_x[f] = c.x;
            f_height_y[f] = c.y;
            f_height_z[f] = c.z;
        }
        progress.next();

        // add a vector field to the faces
        mesh->update_face_normals();
        progress.next();
    }


    model->renderer()->update();
    viewer()->update();
    updateUi();
}


void MainWindow::surfaceMeshExtractConnectedComponents() {
    auto mesh = dynamic_cast<SurfaceMesh*>(viewer_->currentModel());
    if (!mesh)
        return;

    const auto& components = SurfaceMeshComponent::extract(mesh);
    LOG(INFO) << "model has " << components.size() << " connected components";

    const std::string color_name = "f:color_components";
    auto face_color = mesh->face_property<vec3>(color_name, vec3(0.5f, 0.5f, 0.5f));
    for (auto& comp : components) {
        const vec3& color = random_color(false);
        for (auto f : comp.faces())
            face_color[f] = color;
    }

    auto faces = mesh->renderer()->get_triangles_drawable("faces");
    faces->set_property_coloring(State::FACE, color_name);

    mesh->renderer()->update();
    viewer()->update();
    updateUi();
}


void MainWindow::surfaceMeshPlanarPartition() {
    auto mesh = dynamic_cast<SurfaceMesh*>(viewer_->currentModel());
    if (!mesh)
        return;

    const std::string partition_name = "f:planar_partition";
    auto planar_segments = mesh->face_property<int>(partition_name, -1);
    SurfaceMeshEnumerator::enumerate_planar_components(mesh, planar_segments, 1.0f);

    const std::string color_name = "f:color_planar_partition";
    auto coloring = mesh->face_property<vec3>(color_name, vec3(0, 0, 0));
    Renderer::color_from_segmentation(mesh, planar_segments, coloring);
    auto faces = mesh->renderer()->get_triangles_drawable("faces");
    faces->set_property_coloring(State::FACE, color_name);

    mesh->renderer()->update();
    viewer()->update();
    updateUi();
}


void MainWindow::surfaceMeshDual() {
    auto mesh = dynamic_cast<SurfaceMesh*>(viewer_->currentModel());
    if (!mesh)
        return;

    geom::dual(mesh);

    mesh->renderer()->update();
    viewer()->update();
    updateUi();
}


void MainWindow::surfaceMeshPolygonization() {
    auto mesh = dynamic_cast<SurfaceMesh*>(viewer_->currentModel());
    if (!mesh)
        return;

#if HAS_CGAL
    // stitch first: to encourage large polygons
    Surfacer::stitch_borders(mesh);
    Surfacer::merge_reversible_connected_components(mesh);
#endif

    // polygonization
    SurfaceMeshPolygonization polygonizer;
    polygonizer.apply(mesh);

#if HAS_CGAL
    // stitch again (the "merge-edge" edge operation in polygonization may result in some borders)
    Surfacer::stitch_borders(mesh);
    Surfacer::merge_reversible_connected_components(mesh);
#endif

    mesh->renderer()->update();
    viewer()->update();
    updateUi();
}


void MainWindow::surfaceMeshSubdivisionCatmullClark() {
    SurfaceMesh* mesh = dynamic_cast<SurfaceMesh*>(viewer()->currentModel());
    if (!mesh)
        return;

    if (SurfaceMeshSubdivision::catmull_clark(mesh)) {
        mesh->renderer()->update();
        viewer()->update();
        updateUi();
    }
}


void MainWindow::surfaceMeshSubdivisionLoop() {
    SurfaceMesh* mesh = dynamic_cast<SurfaceMesh*>(viewer()->currentModel());
    if (!mesh)
        return;

    if (SurfaceMeshSubdivision::loop(mesh)) {
        mesh->renderer()->update();
        viewer()->update();
        updateUi();
    }
}


void MainWindow::surfaceMeshSubdivisionSqrt3() {
    SurfaceMesh* mesh = dynamic_cast<SurfaceMesh*>(viewer()->currentModel());
    if (!mesh)
        return;

    if (SurfaceMeshSubdivision::sqrt3(mesh)) {
        mesh->renderer()->update();
        viewer()->update();
        updateUi();
    }
}


void MainWindow::manipulateProperties() {
    static DialogProperties* dialog = nullptr;
    if (!dialog)
        dialog = new DialogProperties(this);
    dialog->show();
}


void MainWindow::pointCloudPoissonSurfaceReconstruction() {
    static DialogPoissonReconstruction* dialog = nullptr;
    if (!dialog)
        dialog = new DialogPoissonReconstruction(this);
    dialog->show();
}


void MainWindow::pointCloudRansacPrimitiveExtraction() {
    static DialogPointCloudRansacPrimitiveExtraction* dialog = nullptr;
    if (!dialog)
        dialog = new DialogPointCloudRansacPrimitiveExtraction(this);
    dialog->show();
}


void MainWindow::surfaceMeshSampling() {
    static DialogSurfaceMeshSampling* dialog = nullptr;
    if (!dialog)
        dialog = new DialogSurfaceMeshSampling(this);
    dialog->show();
}


void MainWindow::pointCloudDownsampling() {
    static DialogPointCloudSimplification* dialog = nullptr;
    if (!dialog)
        dialog = new DialogPointCloudSimplification(this);
    dialog->show();
}

namespace details {
    template<typename MODEL>
    void translate(MODEL* model, const vec3& p) {
        auto points = model->template get_vertex_property<vec3>("v:point");
        for (auto v : model->vertices())
            points[v] -= p;
    }
}

void MainWindow::translationalRecenter() {
    Model* first_model = viewer_->models()[0];

    const vec3 origin = first_model->bounding_box().center();
    for (auto model : viewer_->models()) {
        if (dynamic_cast<SurfaceMesh*>(model))
            details::translate(dynamic_cast<SurfaceMesh*>(model), origin);
        else if (dynamic_cast<PointCloud*>(model))
            details::translate(dynamic_cast<PointCloud*>(model), origin);
        else if (dynamic_cast<Graph*>(model))
            details::translate(dynamic_cast<Graph*>(model), origin);
        else if (dynamic_cast<PolyMesh*>(model))
            details::translate(dynamic_cast<PolyMesh*>(model), origin);

        model->manipulator()->reset();
        model->renderer()->update();
    }
    
    // since translated, recenter to screen
    viewer_->fitScreen();
}


void MainWindow::addGaussianNoise() {
    static DialogGaussianNoise* dialog = nullptr;
    if (!dialog)
        dialog = new DialogGaussianNoise(this);
    dialog->show();
}


void MainWindow::applyManipulatedTransformation() {
    Model* model = viewer_->currentModel();
    if (!model)
        return;

    mat4 manip = model->manipulator()->matrix();
    auto& points = model->points();
    for (auto& p : points)
        p = manip * p;

    if (dynamic_cast<SurfaceMesh*>(model)) {
        dynamic_cast<SurfaceMesh *>(model)->update_vertex_normals();
    }
    else if (dynamic_cast<PointCloud*>(model)) {
        PointCloud* cloud = dynamic_cast<PointCloud*>(model);
        auto normal = cloud->get_vertex_property<vec3>("v:normal");
        if (normal) {
            const mat3& N = transform::normal_matrix(manip);
            for (auto v : cloud->vertices())
                normal[v] = N * normal[v];
            // vector fields...
        }
    }

    model->manipulator()->reset();
    model->renderer()->update();
    viewer_->update();
}


void MainWindow::giveUpManipulatedTransformation() {
    Model* model = viewer_->currentModel();
    if (!model)
        return;

    model->manipulator()->reset();
    model->renderer()->update();
    viewer_->update();
}


void MainWindow::computeSurfaceMeshCurvatures() {
    static DialogSurfaceMeshCurvature* dialog = nullptr;
    if (!dialog)
        dialog = new DialogSurfaceMeshCurvature(this);
    dialog->show();
}


void MainWindow::surfaceMeshSimplification() {
    static DialogSurfaceMeshSimplification* dialog = nullptr;
    if (!dialog)
        dialog = new DialogSurfaceMeshSimplification(this);
    dialog->show();
}


void MainWindow::surfaceMeshSmoothing() {
    static DialogSurfaceMeshSmoothing* dialog = nullptr;
    if (!dialog)
        dialog = new DialogSurfaceMeshSmoothing(this);
    dialog->show();
}


void MainWindow::surfaceMeshFairing() {
    static DialogSurfaceMeshFairing* dialog = nullptr;
    if (!dialog)
        dialog = new DialogSurfaceMeshFairing(this);
    dialog->show();
}


void MainWindow::surfaceMeshHoleFilling() {
    static DialogSurfaceMeshHoleFilling* dialog = nullptr;
    if (!dialog)
        dialog = new DialogSurfaceMeshHoleFilling(this);
    dialog->show();
}


void MainWindow::surfaceMeshRemeshing() {
    static DialogSurfaceMeshRemeshing* dialog = nullptr;
    if (!dialog)
        dialog = new DialogSurfaceMeshRemeshing(this);
    dialog->show();
}


void MainWindow::surfaceMeshParameterization() {
    static DialogSurfaceMeshParameterization* dialog = nullptr;
    if (!dialog)
        dialog = new DialogSurfaceMeshParameterization(this);
    dialog->show();
}


void MainWindow::animation() {
    static DialogWalkThrough* dialog = nullptr;
    if (!dialog)
        dialog = new DialogWalkThrough(this);

    viewer_->walkThrough()->set_scene(viewer_->models());

    // don't allow model picking when creating camera paths.
    ui->actionSelectModel->setChecked(false);
    for (auto m : viewer_->models())
        m->renderer()->set_selected(false);

    dialog->show();
}


void MainWindow::showPrimitivePropertyUnderMouse(bool b) {
    if (b)
        ui->actionShowCordinatesUnderMouse->setChecked(false);
    viewer_->showPrimitivePropertyUnderMouse(b);
}


void MainWindow::showCoordinatesUnderMouse(bool b) {
    if (b)
        ui->actionShowPrimitivePropertyUnderMouse->setChecked(false);
    viewer_->showCoordinatesUnderMouse(b);
}


void MainWindow::importCameraPath() {
    std::string dir = "./";
    if (viewer_->currentModel())
        dir = file_system::parent_directory(viewer_->currentModel()->name());
    QString suggested_dir = QString::fromStdString(dir);
    const QString fileName = QFileDialog::getOpenFileName(
            this,
            "Import keyframes from file",
            suggested_dir,
            "Keyframe file (*.kf)\n"
            "All formats (*.*)"
    );

    if (fileName.isEmpty())
        return;

    if (viewer_->walkThrough()->interpolator()->read_keyframes(fileName.toStdString())) {
        LOG(INFO) << viewer_->walkThrough()->interpolator()->number_of_keyframes() << " keyframes loaded";
        viewer_->adjustSceneRadius();
    }

    viewer_->update();
}


void MainWindow::exportCameraPath() {
    if (viewer_->walkThrough()->interpolator()->number_of_keyframes() == 0) {
        LOG(INFO) << "nothing can be exported (path is empty)";
        return;
    }

    std::string name = "./keyframes.kf";
    if (viewer_->currentModel())
        name = file_system::replace_extension(viewer_->currentModel()->name(), "kf");

    QString suggested_name = QString::fromStdString(name);
    const QString fileName = QFileDialog::getSaveFileName(
            this,
            "Export keyframes to file",
            suggested_name,
            "Keyframe file (*.kf)\n"
            "All formats (*.*)"
    );

    if (fileName.isEmpty())
        return;

    if (viewer_->walkThrough()->interpolator()->save_keyframes(fileName.toStdString()))
        LOG(INFO) << "keyframes saved to file";
}


void MainWindow::setShowCameraPath(bool b) {
    viewer_->walkThrough()->set_path_visible(b);
    viewer_->adjustSceneRadius();
    viewer_->update();
}


void MainWindow::setShowKeyframeCameras(bool b) {
    viewer_->walkThrough()->set_cameras_visible(b);
    viewer_->adjustSceneRadius();
    viewer_->update();
}


void MainWindow::surfaceMeshGeodesic() {
    SurfaceMesh *mesh = dynamic_cast<SurfaceMesh *>(viewer()->currentModel());
    if (!mesh)
        return;

    const auto& components = SurfaceMeshComponent::extract(mesh);
    if (components.size() > 1)
        LOG(WARNING) << "mesh has " << components.size() << " connected components. Geodesic computation is valid on a single component";

    // pick a few a random vertices and mark them locked
    auto locked = mesh->vertex_property<bool>("v:locked", false);
    locked.vector().assign(mesh->n_vertices(), false);

    // setup seeds
    std::vector<SurfaceMesh::Vertex> seeds;
    const int num_seeds = 1;
    for (int i=0; i<num_seeds; ++i) {
        const int idx = rand() % mesh->n_vertices();
        SurfaceMesh::Vertex v(idx);
        seeds.push_back(v);
        locked[v] = true;
    }
    LOG(WARNING) << "this only demonstrates Geodesic w.r.t. " << num_seeds << " randomly picked vertices. TODO: ask for user input";

    // compute geodesic distance
    SurfaceMeshGeodesic geodist(mesh);
    geodist.compute(seeds);

    auto drawable = mesh->renderer()->get_triangles_drawable("faces");
    drawable->set_coloring(State::SCALAR_FIELD, State::VERTEX, "v:geodesic:distance");
    mesh->renderer()->update();
    viewer_->update();
    updateRenderingPanel();
}


void MainWindow::pointCloudDelaunayTriangulation2D() {
    auto cloud = dynamic_cast<PointCloud*>(viewer()->currentModel());
    if (!cloud)
        return;

    const std::vector<vec3>& pts = cloud->points();

    std::vector<vec2> points;
    for (std::size_t i = 0; i < pts.size(); ++i) {
        points.push_back(vec2(pts[i]));
    }

    Delaunay2 delaunay;
    delaunay.set_vertices(points);

    SurfaceMesh* mesh = new SurfaceMesh;
    const std::string &name = file_system::name_less_extension(mesh->name()) + "_delaunay_XY.ply";
    mesh->set_name(name);

    for (std::size_t i = 0; i < points.size(); i++) {
        mesh->add_vertex(vec3(points[i], pts[i].z));
    }

    for (unsigned int i = 0; i < delaunay.nb_triangles(); i++) {
        std::vector<SurfaceMesh::Vertex> vts(3);
        for (int j = 0; j < 3; j++) {
            const int v = delaunay.tri_vertex(i, j);
            assert(v >= 0);
            assert(v < points.size());
            vts[j] = SurfaceMesh::Vertex(v);
        }
        mesh->add_face(vts);
    }

    viewer_->addModel(mesh);
    updateUi();
    viewer_->update();
}


void MainWindow::pointCloudDelaunayTriangulation3D() {
    auto cloud = dynamic_cast<PointCloud*>(viewer()->currentModel());
    if (!cloud)
        return;

    const std::vector<vec3>& points = cloud->points();
    Delaunay3 delaunay;
    delaunay.set_vertices(points);

    PolyMesh* mesh = new PolyMesh;
    const std::string &name = file_system::name_less_extension(cloud->name()) + "_delaunay.ply";
    mesh->set_name(name);

    for (std::size_t i = 0; i < points.size(); i++) {
        mesh->add_vertex(points[i]);
    }

    LOG(INFO) << "building tetrahedral mesh with " << delaunay.nb_tets() << " tetrahedra...";
    StopWatch w;
    for (unsigned int i = 0; i < delaunay.nb_tets(); i++) {
        PolyMesh::Vertex vts[4];
        for (int j = 0; j < 4; j++) {
            int v = delaunay.tet_vertex(i, j);
            assert(v >= 0);
            assert(v < points.size());
            vts[j] = PolyMesh::Vertex(v);
        }
        mesh->add_tetra(vts[0], vts[1], vts[2], vts[3]);
    }
    LOG(INFO) << "done. " << w.time_string();

    viewer_->addModel(mesh);
    updateUi();
    viewer_->update();
}