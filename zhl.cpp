// new
// test2
auto models = viewer_->models();
for (auto m : models) {
    if (m->name() == "Plane_plus") {
        viewer_->deleteModel(m);
    }
}
viewer_->addModel(planes_updown[0].mesh);

auto this_borders = this_mesh->renderer()->get_lines_drawable("borders");
this_borders->set_line_width(2.0f);
this_borders->set_uniform_coloring(vec4(1, 0, 0, 1));   
this_borders->set_visible(true);
auto this_faces = this_mesh->renderer()->get_triangles_drawable("faces");
this_faces->set_visible(false);
auto this_vertices = this_mesh->renderer()->get_points_drawable("vertices");
this_vertices->set_uniform_coloring(vec4(1, 1, 0, 0.7));
this_vertices->set_point_size(4.0f);
this_vertices->set_visible(false);

std::vector<Graph::Vertex> g_edges, gg_edges;
std::vector<Graph::Vertex> center_v_edges;

auto models = viewer_->models();
for (auto m : models) {
    if (m->name() == "outcome" || m->name() == "center_v") {
        viewer_->deleteModel(m);
    }
}

Graph* outcome = new Graph; 
Graph* center_v = new Graph; 
outcome->set_name("outcome");
center_v->set_name("center_v");

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

viewer_->addModel(outcome);
viewer_->addModel(center_v);

auto edges = outcome->renderer()->get_lines_drawable("edges");
edges->set_uniform_coloring(vec4(0, 1, 0, 1));
auto vertices = outcome->renderer()->get_points_drawable("vertices");
vertices->set_uniform_coloring(vec4(1, 0, 0, 1));
vertices->set_point_size(10.0f);

// connect caifang
// updateUi();
viewer_->update();

// 20220903
Graph* outcome = new Graph;
outcome->set_name("outcome");
Graph* center_v = new Graph; 
center_v->set_name("center_v");
Graph* Plane_plus = new Graph; 
Plane_plus->set_name("Plane_plus");

outcome->add_vertex(vec3(0));
center_v->add_vertex(vec3(0));
Plane_plus->add_vertex(vec3(0));

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

std::vector<vec3> alpha_points;
std::vector<unsigned int> alpha_indices;
alpha_points.push_back(p0);
alpha_points.push_back(p1);

auto models = viewer_->models();
for (auto m : models) {
    if (m->name() == "outcome") {
        auto drawable_points = m->renderer()->get_points_drawable("vertices");
        auto drawable_lines  = m->renderer()->get_lines_drawable("edges");
        static std::vector<unsigned int> wire_indices = {0, 1, 1, 2, 2, 3, 3, 0, 4, 5,
                                                         6, 7, 7, 8, 8, 9, 9, 6, 10, 11};
        std::vector<vec3> points;
        points.insert(points.end(), cgal_rect.begin(), cgal_rect.end());
        points.insert(points.end(), cgal_rect_project.begin(), cgal_rect_project.end());
        viewer_->makeCurrent();

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

        drawable_points->update_vertex_buffer(alpha_points, true);
        drawable_lines->update_vertex_buffer(alpha_points, true);
        drawable_lines->update_element_buffer(alpha_indices);

        viewer_->doneCurrent(); 
    }
}

viewer_->update();


// 20220905
SurfaceMesh *Planar_segment::borders_alpha_plus(float dd1, float dd2, const Plane3 *plane_lidar) {
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

    float avg_spacing = 0.03;
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

SurfaceMesh *borders_alpha_plus(float dd1, float dd2, const Plane3 *plane_lidar);

// 20220907
cgal_line_threshold: 0.1
float cgal_line_threshold = params_["cgal_line_threshold"].As<float> ();

alpha_spacing: 0.03


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

//
        if (size_updown >=2 && size_topbot >=1) {
        if (size_updown >=2 && size_topbot ==0 && size_leftri ==2) {

cgal_plane_bottom -> distance_tail()

when parallel find okk

//with celang
if (size_leftri == 2) {
use_leftri: false


if (size_leftri >=2)
{
    vec3 s(0, 0, 0);
    vec3 t(10000, 0, 0);
    vec3 p;
    bool k = planes_leftri[0].plane->intersect(s, t, p); 
    if (k)
        std::swap(planes_leftri[0], planes_leftri[1]);
}
