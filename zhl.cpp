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
