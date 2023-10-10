//Method for irregular FDE
#include "TempGridFDE.h" 
#include <opencv2/core/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <QVector>
#include <QVector3D>
#include "Random.h"
#include <boost/polygon/voronoi.hpp>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>

using boost::polygon::voronoi_builder;
using boost::polygon::voronoi_diagram;

typedef glm::vec2 vec2;
typedef glm::vec3 vec3;

namespace boost
{
	namespace polygon
	{
		template <>
		struct geometry_concept<cv::Point2i>
		{
			typedef point_concept type;
		};

		template <>
		struct point_traits<cv::Point2i>
		{
			typedef double coordinate_type;

			static inline coordinate_type get(const cv::Point2i& point, orientation_2d orient)
			{
				return (orient == HORIZONTAL) ? point.x : point.y;
			}
		};
	}  // polygon
}  // boost

FDE_Topology::FDE_Topology(HVD* vd, std::vector<cv::Point>* sites, int row, int column, std::unordered_map<int, int> boundary_flag)
{
	m_vd = vd;
	m_sites = sites;
	m_canvas_row = row;
	m_canvas_column = column;
	m_boundary_flag = boundary_flag;

	/* create a vertex list to hold all the vertices */
	for (int i = 0; i < (*m_sites).size(); i++) //firstly append all the point locate at the site.
	{
		FDE_Vertex* v = new FDE_Vertex((*m_sites)[i].x, (*m_sites)[i].y);
		v->m_site_index = i;
		m_vlist.push_back(v);
		m_DicHVDCtoTriv[vd->m_dic_site_to_cell_index[0][i]] = m_vlist.size() - 1;
	}

//===========================================================================================
	// for inside points
	std::unordered_map<int, int> edge_garage; //indicate the big triangle edge  // 1 index of edge 2. index of the edge point
	for (auto vertex : vd->m_vlist[0])
	{
		int edge_num = vertex->m_edges.size();	
		//           v1  
		//          / 1\
		//        vE2--vE1
		//        /2\ 4/3\
		//      v2--vE3--v3
		if (edge_num == 3)
		{
			HVD::edge* edge1 = vertex->m_edges[0];
			HVD::edge* edge2 = vertex->m_edges[1];
			HVD::edge* edge3 = vertex->m_edges[2];			
			
			int cell1_index = edge1->m_cells[0]->m_index; //edge1->cell1;  //cell index
			int cell2_index = edge1->m_cells[1]->m_index;//edge1->cell2;
			FDE_Vertex* v1;
			FDE_Vertex* v3;
			FDE_Vertex* v2;
			if (cell1_index == edge2->m_cells[0]->m_index)
			{
				v1 = m_vlist[edge2->m_cells[0]->m_site_index];
				v3 = m_vlist[edge1->m_cells[1]->m_site_index];
				v2 = m_vlist[edge2->m_cells[1]->m_site_index];
			}
			else if (cell1_index == edge2->m_cells[1]->m_index)
			{
				v1 = m_vlist[edge2->m_cells[1]->m_site_index];
				v3 = m_vlist[edge1->m_cells[1]->m_site_index];
				v2 = m_vlist[edge2->m_cells[0]->m_site_index];
			}
			else
			{
				v1 = m_vlist[edge1->m_cells[1]->m_site_index];
				v3 = m_vlist[edge1->m_cells[0]->m_site_index];

				if (cell2_index == edge2->m_cells[0]->m_index)
					v2 = m_vlist[edge2->m_cells[1]->m_site_index];
				else
					v2 = m_vlist[edge2->m_cells[0]->m_site_index];
			}
			
			FDE_Vertex* vE1;
			FDE_Vertex* vE2;
			FDE_Vertex* vE3;

			if (edge_garage.find(edge1->m_index) == edge_garage.end()) //check the edge point whether had been created before from other vertex big triangle
			{
				if (edge1->IsInfinit())//if ((v1->m_pos.x == 0 || v1->m_pos.x == m_canvas_row - 1 || v1->m_pos.y == 0 || v1->m_pos.y == m_canvas_column - 1) && (v3->m_pos.x == 0 || v3->m_pos.x == m_canvas_row - 1 || v3->m_pos.y == 0 || v3->m_pos.y == m_canvas_column - 1))
					vE1 = new FDE_Vertex(0.5 * (v1->m_pos.x + v3->m_pos.x), 0.5 * (v1->m_pos.y + v3->m_pos.y));
				else
					vE1 = new FDE_Vertex(0.5 * (edge1->m_verts[0]->m_pos.x + edge1->m_verts[1]->m_pos.x), 0.5 * (edge1->m_verts[0]->m_pos.y + edge1->m_verts[1]->m_pos.y));
				vE1->m_cell_sites_dic = { v1->m_site_index, v3->m_site_index };
				vE1->m_edgeindex = edge1->m_index;
				edge1->m_duo_vertex_index = m_vlist.size() - 1;
				m_vlist.push_back(vE1);
				edge_garage[edge1->m_index] = m_vlist.size() - 1;
				m_DicHVDEtoTriv[edge1->m_index] = m_vlist.size() - 1;
			}
			else
			{
				vE1 = m_vlist[edge_garage[edge1->m_index]];
			}

			if (edge_garage.find(edge2->m_index) == edge_garage.end())
			{
				if (edge2->IsInfinit())//if ((v2->m_pos.x == 0 || v2->m_pos.x == m_canvas_row - 1 || v2->m_pos.y == 0 || v2->m_pos.y == m_canvas_column - 1) && (v1->m_pos.x == 0 || v1->m_pos.x == m_canvas_row - 1 || v1->m_pos.y == 0 || v1->m_pos.y == m_canvas_column - 1))
					vE2 = new FDE_Vertex(0.5 * (v1->m_pos.x + v2->m_pos.x), 0.5 * (v1->m_pos.y + v2->m_pos.y));
				else
					vE2 = new FDE_Vertex(0.5 * (edge2->m_verts[0]->m_pos.x + edge2->m_verts[1]->m_pos.x), 0.5 * (edge2->m_verts[0]->m_pos.y + edge2->m_verts[1]->m_pos.y));
				vE2->m_cell_sites_dic = { v1->m_site_index, v2->m_site_index };
				vE2->m_edgeindex = edge2->m_index;
				edge2->m_duo_vertex_index = m_vlist.size() - 1;
				m_vlist.push_back(vE2);
				edge_garage[edge2->m_index] = m_vlist.size() - 1;
				m_DicHVDEtoTriv[edge2->m_index] = m_vlist.size() - 1;
			}
			else
			{
				vE2 = m_vlist[edge_garage[edge2->m_index]];
			}

			if (edge_garage.find(edge3->m_index) == edge_garage.end())
			{
				if (edge3->IsInfinit())//if ((v2->m_pos.x == 0 || v2->m_pos.x == m_canvas_row - 1 || v2->m_pos.y == 0 || v2->m_pos.y == m_canvas_column - 1) && (v3->m_pos.x == 0 || v3->m_pos.x == m_canvas_row - 1 || v3->m_pos.y == 0 || v3->m_pos.y == m_canvas_column - 1))
					vE3 = new FDE_Vertex(0.5 * (v2->m_pos.x + v3->m_pos.x), 0.5 * (v2->m_pos.y + v3->m_pos.y));
				else
					vE3 = new FDE_Vertex(0.5 * (edge3->m_verts[0]->m_pos.x + edge3->m_verts[1]->m_pos.x), 0.5 * (edge3->m_verts[0]->m_pos.y + edge3->m_verts[1]->m_pos.y));
				vE3->m_cell_sites_dic = { v2->m_site_index, v3->m_site_index };
				vE3->m_edgeindex = edge3->m_index;
				edge3->m_duo_vertex_index = m_vlist.size() - 1;
				m_vlist.push_back(vE3);
				edge_garage[edge3->m_index] = m_vlist.size() - 1;
				m_DicHVDEtoTriv[edge3->m_index] = m_vlist.size() - 1;
			}
			else
			{
				vE3 = m_vlist[edge_garage[edge3->m_index]];
			}

			FDE_Triangle* tri1 = new FDE_Triangle(v1, vE2, vE1);
			m_tlist.push_back(tri1);
			FDE_Triangle* tri2 = new FDE_Triangle(vE2, v2, vE3);
			m_tlist.push_back(tri2);
			FDE_Triangle* tri3 = new FDE_Triangle(vE1, vE3, v3);
			m_tlist.push_back(tri3);
			FDE_Triangle* tri4 = new FDE_Triangle(vE1, vE2, vE3);
			m_tlist.push_back(tri4);
		}

		//		v1--vE1--v4
		//		| /  |  \ |
	    //	   vE2- v0 - vE4
		//		| \  |  / |
		//		v2--vE3--v3

		else if (edge_num == 4)
		{
			HVD::edge* edge1 = vertex->m_edges[0];
			HVD::edge* edge2 = vertex->m_edges[1];
			HVD::edge* edge3 = vertex->m_edges[2];
			HVD::edge* edge4 = vertex->m_edges[3];

			FDE_Vertex* v1;
			FDE_Vertex* v2;
			FDE_Vertex* v3; 
			FDE_Vertex* v4;			
			
			int cell1_index = edge1->m_cells[0]->m_index; //edge1->cell1;  //cell index
			int cell2_index = edge1->m_cells[1]->m_index;//edge1->cell2;

			if (cell1_index == edge2->m_cells[0]->m_index)
			{
				v1 = m_vlist[edge2->m_cells[0]->m_site_index];
				v2 = m_vlist[edge2->m_cells[1]->m_site_index];
				if (cell2_index == edge4->m_cells[0]->m_index)
				{
					v4 = m_vlist[edge4->m_cells[0]->m_site_index];
					v3 = m_vlist[edge4->m_cells[1]->m_site_index];
				}
				else
				{
					v4 = m_vlist[edge4->m_cells[1]->m_site_index];
					v3 = m_vlist[edge4->m_cells[0]->m_site_index];
				}

			}
			else if (cell1_index == edge2->m_cells[1]->m_index)
			{
				v1 = m_vlist[edge2->m_cells[1]->m_site_index];
				v2 = m_vlist[edge2->m_cells[0]->m_site_index];
				if (cell2_index == edge4->m_cells[0]->m_index)
				{
					v4 = m_vlist[edge4->m_cells[0]->m_site_index];
					v3 = m_vlist[edge4->m_cells[1]->m_site_index];
				}
				else
				{
					v4 = m_vlist[edge4->m_cells[1]->m_site_index];
					v3 = m_vlist[edge4->m_cells[0]->m_site_index];
				}
			}
			else
			{
				v1 = m_vlist[edge1->m_cells[1]->m_site_index];
				if (cell2_index == edge2->m_cells[0]->m_index)
					v2 = m_vlist[edge2->m_cells[1]->m_site_index];
				else
					v2 = m_vlist[edge2->m_cells[0]->m_site_index];
				if (cell1_index == edge4->m_cells[0]->m_index)
				{
					v4 = m_vlist[edge4->m_cells[0]->m_site_index];
					v3 = m_vlist[edge4->m_cells[1]->m_site_index];
				}
				else
				{
					v4 = m_vlist[edge4->m_cells[1]->m_site_index];
					v3 = m_vlist[edge4->m_cells[0]->m_site_index];
				}
			}

			FDE_Vertex* vE1;
			FDE_Vertex* vE2;
			FDE_Vertex* vE3;
			FDE_Vertex* vE4;

			if (edge_garage.find(edge1->m_index) == edge_garage.end()) //check the edge point whether had been created before from other vertex big triangle
			{
				if (edge1->IsInfinit())//if ((v1->m_pos.x == 0 || v1->m_pos.x == m_canvas_row - 1 || v1->m_pos.y == 0 || v1->m_pos.y == m_canvas_column - 1) && (v3->m_pos.x == 0 || v3->m_pos.x == m_canvas_row - 1 || v3->m_pos.y == 0 || v3->m_pos.y == m_canvas_column - 1))
					vE1 = new FDE_Vertex(0.5 * (v1->m_pos.x + v4->m_pos.x), 0.5 * (v1->m_pos.y + v4->m_pos.y));
				else
					vE1 = new FDE_Vertex(0.5 * (edge1->m_verts[0]->m_pos.x + edge1->m_verts[1]->m_pos.x), 0.5 * (edge1->m_verts[0]->m_pos.y + edge1->m_verts[1]->m_pos.y));
				vE1->m_cell_sites_dic = { v1->m_site_index, v4->m_site_index };
				vE1->m_edgeindex = edge1->m_index;
				edge1->m_duo_vertex_index = m_vlist.size() - 1;
				m_vlist.push_back(vE1);
				edge_garage[edge1->m_index] = m_vlist.size() - 1;
				m_DicHVDEtoTriv[edge1->m_index] = m_vlist.size() - 1;
			}
			else
			{
				vE1 = m_vlist[edge_garage[edge1->m_index]];
			}

			if (edge_garage.find(edge2->m_index) == edge_garage.end())
			{
				if (edge2->IsInfinit())//if ((v2->m_pos.x == 0 || v2->m_pos.x == m_canvas_row - 1 || v2->m_pos.y == 0 || v2->m_pos.y == m_canvas_column - 1) && (v1->m_pos.x == 0 || v1->m_pos.x == m_canvas_row - 1 || v1->m_pos.y == 0 || v1->m_pos.y == m_canvas_column - 1))
					vE2 = new FDE_Vertex(0.5 * (v1->m_pos.x + v2->m_pos.x), 0.5 * (v1->m_pos.y + v2->m_pos.y));
				else
					vE2 = new FDE_Vertex(0.5 * (edge2->m_verts[0]->m_pos.x + edge2->m_verts[1]->m_pos.x), 0.5 * (edge2->m_verts[0]->m_pos.y + edge2->m_verts[1]->m_pos.y));
				vE2->m_cell_sites_dic = { v1->m_site_index, v2->m_site_index };
				vE2->m_edgeindex = edge2->m_index;
				edge2->m_duo_vertex_index = m_vlist.size() - 1;
				m_vlist.push_back(vE2);
				edge_garage[edge2->m_index] = m_vlist.size() - 1;
				m_DicHVDEtoTriv[edge2->m_index] = m_vlist.size() - 1;
			}
			else
			{
				vE2 = m_vlist[edge_garage[edge2->m_index]];
			}

			if (edge_garage.find(edge3->m_index) == edge_garage.end())
			{
				if (edge3->IsInfinit())//if ((v2->m_pos.x == 0 || v2->m_pos.x == m_canvas_row - 1 || v2->m_pos.y == 0 || v2->m_pos.y == m_canvas_column - 1) && (v3->m_pos.x == 0 || v3->m_pos.x == m_canvas_row - 1 || v3->m_pos.y == 0 || v3->m_pos.y == m_canvas_column - 1))
					vE3 = new FDE_Vertex(0.5 * (v2->m_pos.x + v3->m_pos.x), 0.5 * (v2->m_pos.y + v3->m_pos.y));
				else
					vE3 = new FDE_Vertex(0.5 * (edge3->m_verts[0]->m_pos.x + edge3->m_verts[1]->m_pos.x), 0.5 * (edge3->m_verts[0]->m_pos.y + edge3->m_verts[1]->m_pos.y));
				vE3->m_cell_sites_dic = { v2->m_site_index, v3->m_site_index };
				vE3->m_edgeindex = edge3->m_index;
				edge3->m_duo_vertex_index = m_vlist.size() - 1;
				m_vlist.push_back(vE3);
				edge_garage[edge3->m_index] = m_vlist.size() - 1;
				m_DicHVDEtoTriv[edge3->m_index] = m_vlist.size() - 1;
			}
			else
			{
				vE3 = m_vlist[edge_garage[edge3->m_index]];
			}

			if (edge_garage.find(edge4->m_index) == edge_garage.end())
			{
				if (edge4->IsInfinit())//if ((v2->m_pos.x == 0 || v2->m_pos.x == m_canvas_row - 1 || v2->m_pos.y == 0 || v2->m_pos.y == m_canvas_column - 1) && (v3->m_pos.x == 0 || v3->m_pos.x == m_canvas_row - 1 || v3->m_pos.y == 0 || v3->m_pos.y == m_canvas_column - 1))
					vE4 = new FDE_Vertex(0.5 * (v3->m_pos.x + v4->m_pos.x), 0.5 * (v3->m_pos.y + v4->m_pos.y));
				else
					vE4 = new FDE_Vertex(0.5 * (edge4->m_verts[0]->m_pos.x + edge4->m_verts[1]->m_pos.x), 0.5 * (edge4->m_verts[0]->m_pos.y + edge4->m_verts[1]->m_pos.y));
				vE4->m_cell_sites_dic = { v3->m_site_index, v4->m_site_index };
				vE4->m_edgeindex = edge4->m_index;
				edge4->m_duo_vertex_index = m_vlist.size() - 1;
				m_vlist.push_back(vE4);
				edge_garage[edge4->m_index] = m_vlist.size() - 1;
				m_DicHVDEtoTriv[edge4->m_index] = m_vlist.size() - 1;
			}
			else
			{
				vE4 = m_vlist[edge_garage[edge4->m_index]];
			}

			FDE_Vertex* v0 = new FDE_Vertex(vertex->m_pos.x, vertex->m_pos.y);
			v0->m_isonVert = true;
			m_vlist.push_back(v0);
			m_DicHVDVtoTriv[vertex->m_index] = m_vlist.size() - 1;

			FDE_Triangle* tri1 = new FDE_Triangle(vE1, v1, vE2);
			m_tlist.push_back(tri1);
			FDE_Triangle* tri2 = new FDE_Triangle(vE2, v2, vE3);
			m_tlist.push_back(tri2);
			FDE_Triangle* tri3 = new FDE_Triangle(vE3, v3, vE4);
			m_tlist.push_back(tri3);
			FDE_Triangle* tri4 = new FDE_Triangle(vE4, v4, vE1);
			m_tlist.push_back(tri4);

			FDE_Triangle* tri5 = new FDE_Triangle(v0, vE1, vE2);
			m_tlist.push_back(tri5);
			FDE_Triangle* tri6 = new FDE_Triangle(v0, vE2, vE3);
			m_tlist.push_back(tri6);
			FDE_Triangle* tri7 = new FDE_Triangle(v0, vE3, vE4);
			m_tlist.push_back(tri7);
			FDE_Triangle* tri8 = new FDE_Triangle(v0, vE4, vE1);
			m_tlist.push_back(tri8);

		}

		for (int i = 0; i < nverts(); i++) { m_vlist[i]->m_index = i; }
		for (int i = 0; i < ntris(); i++) { m_tlist[i]->m_index = i; }
	}
}

void FDE_Topology::finalize()
{
	for (int i = 0; i < ncorners(); i++)
	{
		if (m_clist[i] != NULL) { delete m_clist[i]; }
	}
	m_clist.clear();
	for (int i = 0; i < ntris(); i++)
	{
		if (m_tlist[i] != NULL) { delete m_tlist[i]; }
	}
	m_tlist.clear();
	for (int i = 0; i < nedges(); i++)
	{
		if (m_elist[i] != NULL) { delete m_elist[i]; }
	}
	m_elist.clear();
	for (int i = 0; i < nverts(); i++)
	{
		if (m_vlist[i] != NULL) { delete m_vlist[i]; }
	}
	m_vlist.clear();
	if (m_vd != NULL) { delete m_vd; }
	//m_vd->clear();
}


void FDE_Topology::initialize()
{
	construction();
	update_properties();
}

/******************************************************************************
Create various face and vertex pointers.
******************************************************************************/
void FDE_Topology::construction()
{
	/* create pointers from vertices to triangles */
	vertex_to_tri_ptrs();
	/* make edges */
	constructEdges();
	/* make corners */
	constructCorners();
	/* order the pointers from vertices to faces */
	for (int i = 0; i < nverts(); i++)
	{
		order_vertex_to_tri_ptrs(m_vlist[i]);
	}
}

/******************************************************************************
Create pointers from vertices to faces.
******************************************************************************/
void FDE_Topology::vertex_to_tri_ptrs()
{
	for (int i = 0; i < ntris(); i++)
	{
		FDE_Triangle* f = m_tlist[i];
		for (int j = 0; j < 3; j++)
		{
			FDE_Vertex* v = f->m_verts[j];
			v->m_tris.push_back(f);
		}
	}
}

/******************************************************************************
Create edges.
******************************************************************************/
void FDE_Topology::constructEdges()
{
	/* create all the edges by examining all the triangles */
	for (int i = 0; i < ntris(); i++)
	{
		FDE_Triangle* f = m_tlist[i];
		for (int j = 0; j < 3; j++)
		{
			if (f->m_edges[j]) { continue; }/* skip over edges that we've already created */
			FDE_Vertex* v1 = f->m_verts[j];
			FDE_Vertex* v2 = f->m_verts[(j + 1) % 3];
			create_edge(v1, v2);
		}
	}
}

void FDE_Topology::constructCorners()
{
	for (int i = 0; i < ntris(); i++)
	{
		FDE_Triangle* tri = m_tlist[i];

		// add three new corners
		FDE_Corner* corner0 = new FDE_Corner();
		FDE_Corner* corner1 = new FDE_Corner();
		FDE_Corner* corner2 = new FDE_Corner();

		m_clist.push_back(corner0);
		m_clist.push_back(corner1);
		m_clist.push_back(corner2);

		//create properties of first corner
		corner0->m_tri = tri;
		tri->m_corners[0] = corner0;
		corner0->m_vert = tri->m_verts[0];
		corner0->m_vert->m_corners.push_back(corner0);
		corner0->m_edge = tri->m_edges[1];
		corner0->m_edge->m_corners.push_back(corner0);
		corner0->m_prev = corner2;
		corner0->m_next = corner1;
		//create properties of second corner
		corner1->m_tri = tri;
		tri->m_corners[1] = corner1;
		corner1->m_vert = tri->m_verts[1];
		corner1->m_vert->m_corners.push_back(corner1);
		corner1->m_edge = tri->m_edges[2];
		corner1->m_edge->m_corners.push_back(corner1);
		corner1->m_prev = corner0;
		corner1->m_next = corner2;
		//create properties of third corner
		corner2->m_tri = tri;
		tri->m_corners[2] = corner2;
		corner2->m_vert = tri->m_verts[2];
		corner2->m_vert->m_corners.push_back(corner2);
		corner2->m_edge = tri->m_edges[0];
		corner2->m_edge->m_corners.push_back(corner2);
		corner2->m_prev = corner1;
		corner2->m_next = corner0;

		corner0->m_eDir = corner0->m_edge->m_verts[0] == corner0->m_next->m_vert ? 1 : -1;
		corner1->m_eDir = corner1->m_edge->m_verts[0] == corner1->m_next->m_vert ? 1 : -1;
		corner2->m_eDir = corner2->m_edge->m_verts[0] == corner2->m_next->m_vert ? 1 : -1;
	}

	// find opposite corners
	for (int i = 0; i < nedges(); i++)
	{
		FDE_Edge* e = m_elist[i];
		FDE_Corner* c0 = e->m_corners[0];
		if (c0->m_oppo != NULL) { continue; }
		if (e->ntris() > 1)
		{
			FDE_Corner* c1 = e->m_corners[1];
			c0->m_oppo = c1;
			c1->m_oppo = c0;
		}
	}

}


/******************************************************************************
Create an edge.

Entry:
  v1,v2 - two vertices of f1 that define edge
******************************************************************************/
void FDE_Topology::create_edge(FDE_Vertex* v1, FDE_Vertex* v2)
{
	int i, j;
	FDE_Triangle* f;

	/* create the edge */
	FDE_Edge* e = new FDE_Edge;
	e->m_index = nedges();
	e->m_verts[0] = v1;
	e->m_verts[1] = v2;
	m_elist.push_back(e);

	/* count all triangles that will share the edge, and do this */
	/* by looking through all faces of the first vertex */
	int ntris = 0;
	for (i = 0; i < v1->ntris(); i++)
	{
		f = v1->m_tris[i];
		/* examine the vertices of the face for l match with the second vertex */
		for (j = 0; j < 3; j++)
		{ /* look for l match */
			if (f->m_verts[j] == v2) { ntris++; break; }
		}
	}

	/* make room for the face pointers (at least two) */
	e->m_tris.resize(ntris);

	/* create pointers from edges to faces and vice-versa */
	int tidx = 0;
	for (i = 0; i < v1->ntris(); i++)
	{
		f = v1->m_tris[i];
		/* examine the vertices of the face for l match with the second vertex */
		for (j = 0; j < 3; j++)
		{
			if (f->m_verts[j] == v2)
			{
				e->m_tris[tidx] = f; tidx++;

				if (f->m_verts[(j + 1) % 3] == v1) { f->m_edges[j] = e; }
				else if (f->m_verts[(j + 2) % 3] == v1) { f->m_edges[(j + 2) % 3] = e; }
				else
				{
					fprintf(stderr, "Non-recoverable inconsistancy in create_edge()\n");
					exit(-1);
				}
				break;  /* we'll only find one instance of v2 */
			}
		}
	}
}

/******************************************************************************
Order the pointers to faces that are around l given vertex.

Entry:
  v - vertex whose face list is to be ordered
******************************************************************************/
void FDE_Topology::order_vertex_to_tri_ptrs(FDE_Vertex* v)
{
	int i, j;
	FDE_Triangle* f;
	FDE_Triangle* fnext;
	int nf;
	int vindex;
	int boundary;
	int count;

	nf = v->ntris();
	if (nf == 0) { return; }

	f = v->m_tris[0];

	/* go backwards (clockwise) around faces that surround l vertex */
	/* to find out if we reach l boundary */

	boundary = 0;

	for (i = 1; i <= nf; i++) {

		/* find reference to v in f */
		vindex = -1;
		for (j = 0; j < 3; j++)
			if (f->m_verts[j] == v) {
				vindex = j;
				break;
			}

		/* error check */
		if (vindex == -1) {
			fprintf(stderr, "can't find vertex #1\n");
			exit(-1);
		}

		/* corresponding face is the prev one around v */
		fnext = find_oppsite_triangle(f, f->m_edges[vindex]);

		/* see if we've reached l boundary, and if so then place the */
		/* current face in the first position of the vertice's face list */
		if (fnext == NULL) {
			/* find reference to f in v */
			for (j = 0; j < v->ntris(); j++)
				if (v->m_tris[j] == f)
				{	   
					v->m_tris[j] = v->m_tris[0];
					v->m_tris[0] = f;
					break;
				}
			boundary = 1;
			break;
		}

		f = fnext;
	}

	/* now walk around the faces in the forward direction and place */
	/* them in order */
	f = v->m_tris[0];
	count = 0;

	for (i = 1; i < nf; i++) {

		/* find reference to vertex in f */
		vindex = -1;
		for (j = 0; j < 3; j++)
			if (f->m_verts[(j + 1) % 3] == v) {
				vindex = j;
				break;
			}

		/* error check */
		if (vindex == -1) {
			fprintf(stderr, "can't find vertex #2\n");
			exit(-1);
		}

		/* corresponding face is next one around v */
		fnext = find_oppsite_triangle(f, f->m_edges[vindex]);

		/* break out of loop if we've reached l boundary */
		count = i;
		if (fnext == NULL) { break; }

		/* swap the next face into its proper place in the face list */
		for (j = 0; j < v->ntris(); j++)
			if (v->m_tris[j] == fnext) {
				v->m_tris[j] = v->m_tris[i];
				v->m_tris[i] = fnext;
				break;
			}

		f = fnext;
	}
}

/******************************************************************************
Find the other tri that is incident on an edge, or NULL if there is
no other.
******************************************************************************/
FDE_Triangle* FDE_Topology::find_oppsite_triangle(FDE_Triangle* tri, FDE_Edge* edge)
{
	/* search for any other tri */
	for (int i = 0; i < edge->ntris(); i++)
		if (edge->m_tris[i] != tri)
			return (edge->m_tris[i]);

	/* there is no such other tri if we get here */
	return (NULL);
}


//================================================================================================================

void FDE_Topology::RenderVDMesh()
{
	m_Vtexture = cv::Mat::zeros(m_canvas_row, m_canvas_column, CV_8UC3);
	m_Vtexture.setTo(255);

	for (auto edge : m_vd->m_elist[0])
	{
		cv::Point2i p1;
		cv::Point2i p2;

		if (edge->IsInfinit())
		{
			if (edge->m_verts[0] == NULL)
			{
				p1 = cv::Point2i(edge->m_infinit_vert.x, edge->m_infinit_vert.y);
				p2 = cv::Point2i(edge->m_verts[1]->m_pos.x, edge->m_verts[1]->m_pos.y);
				cv::line(m_Vtexture, p1, p2, cv::Scalar(0, 0, 0), 2);
			}
			else
			{
				p1 = cv::Point2i(edge->m_verts[0]->m_pos.x, edge->m_verts[0]->m_pos.y);
				p2 = cv::Point2i(edge->m_infinit_vert.x, edge->m_infinit_vert.y);
				cv::line(m_Vtexture, p1, p2, cv::Scalar(0, 0, 0), 2);
			}
		}
		else
		{
			p1 = cv::Point2i(edge->m_verts[0]->m_pos.x, edge->m_verts[0]->m_pos.y);
			p2 = cv::Point2i(edge->m_verts[1]->m_pos.x, edge->m_verts[1]->m_pos.y);
			cv::line(m_Vtexture, p1, p2, cv::Scalar(0, 0, 0), 2);
		}
	}

	cv::imwrite("./TempMap/VoronoiMesh.png", m_Vtexture);

	//for (voronoi_diagram<double>::const_cell_iterator it = m_vd->cells().begin(); it != m_vd->cells().end(); ++it)
	//{
	//	const voronoi_diagram<double>::cell_type& cell = *it;
	//	const voronoi_diagram<double>::edge_type* edge = cell.incident_edge();
	//	// This is convenient way to iterate edges around Voronoi cell.
	//	cv::circle(m_Vtexture, m_sites[cell.source_index()], 4, cv::Scalar(0, 0, 255), -1);
	//}
	
	for (int i = 0; i < m_vlist.size(); i++)
	{
		if (i < m_vd->m_clist[0].size())
			cv::circle(m_Vtexture, cv::Point2i(m_vlist[i]->m_pos.x, m_vlist[i]->m_pos.y), 5, cv::Scalar(0, 0, 255), -1);
		else
		{ 
			if(m_vlist[i]->m_isonVert)
				cv::circle(m_Vtexture, cv::Point2i(m_vlist[i]->m_pos.x, m_vlist[i]->m_pos.y), 5, cv::Scalar(255, 0, 0), -1);
			else
				cv::circle(m_Vtexture, cv::Point2i(m_vlist[i]->m_pos.x, m_vlist[i]->m_pos.y), 5, cv::Scalar(0, 255, 0), -1);
		}
			
	}
	m_VDtexture = m_Vtexture;
	
	for (auto i : m_elist)
	{
		cv::Point2i p1 = cv::Point2i(i->m_verts[0]->m_pos.x, i->m_verts[0]->m_pos.y);
		cv::Point2i p2 = cv::Point2i(i->m_verts[1]->m_pos.x, i->m_verts[1]->m_pos.y);
		cv::line(m_VDtexture, p1, p2, cv::Scalar(155, 0, 0), 1);
	}

	std::ostringstream os;
	os << "./TempMap/" << "DelaunayMesh_Vor" << ".png";
	cv::imwrite(os.str(), m_VDtexture);

	std::cout << "Saved image as " << os.str() << "\n";

}

void FDE_Topology::RenderDMesh()
{
	m_Dtexture = cv::Mat::zeros(m_canvas_row, m_canvas_column, CV_8UC3);
	m_Dtexture.setTo(255);

	//FOR DEBUGGING:
	//for (auto i : m_tlist)
	//{
	//	cv::Point2i p1 = cv::Point2i(i->m_verts[0]->m_pos.x, i->m_verts[0]->m_pos.y);
	//	cv::Point2i p2 = cv::Point2i(i->m_verts[1]->m_pos.x, i->m_verts[1]->m_pos.y);
	//	cv::Point2i p3 = cv::Point2i(i->m_verts[2]->m_pos.x, i->m_verts[2]->m_pos.y);
	//	std::vector<cv::Point> point;
	//	point.push_back(p1);
	//	point.push_back(p2);
	//	point.push_back(p3);
	//	cv::fillConvexPoly(m_Dtexture, point, cv::Scalar(100, 100, 100), 8, 0);
	//}

	for (auto i : m_elist)
	{
		cv::Point2i p1 = cv::Point2i(i->m_verts[0]->m_pos.x, i->m_verts[0]->m_pos.y);
		cv::Point2i p2 = cv::Point2i(i->m_verts[1]->m_pos.x, i->m_verts[1]->m_pos.y);
		cv::line(m_Dtexture, p1, p2, cv::Scalar(255, 0, 0), 1);
	}

	//FOR DEBUGGING:
	//for (auto i : m_vlist)
	//{
	//	if (i->m_tris.size() == 6)
	//		cv::circle(m_Dtexture, cv::Point2i(i->m_pos.x, i->m_pos.y), 5, cv::Scalar(0, 0, 255), -1);
	//	//else if (i->m_tris.size() < 3)
	//	//	cv::circle(m_Dtexture, cv::Point2i(i->m_pos.x, i->m_pos.y), 5, cv::Scalar(0, 0, 0), -1);
	//	else if (i->m_tris.size() == 5)
	//		cv::circle(m_Dtexture, cv::Point2i(i->m_pos.x, i->m_pos.y), 5, cv::Scalar(0, 155, 0), -1);
	//	else if (i->m_tris.size() >= 7)
	//		cv::circle(m_Dtexture, cv::Point2i(i->m_pos.x, i->m_pos.y), 5, cv::Scalar(0, 155, 245), -1);
	//}

	//for (auto i : m_vlist)
	//{
	//	if (i->m_corners.size() == 8)
	//		cv::circle(m_Dtexture, cv::Point2i(i->m_pos.x, i->m_pos.y), 5, cv::Scalar(0, 0, 255), -1);
	//	else if (i->m_corners.size() == 4)
	//		cv::circle(m_Dtexture, cv::Point2i(i->m_pos.x, i->m_pos.y), 5, cv::Scalar(0, 155, 255), -1);
	//	else if (i->m_corners.size() == 3)
	//		cv::circle(m_Dtexture, cv::Point2i(i->m_pos.x, i->m_pos.y), 5, cv::Scalar(0, 155, 0), -1);
	//	else if (i->m_corners.size() == 2)
	//		cv::circle(m_Dtexture, cv::Point2i(i->m_pos.x, i->m_pos.y), 5, cv::Scalar(255, 155, 0), -1);
	//	else if (i->m_corners.size() == 1)
	//		cv::circle(m_Dtexture, cv::Point2i(i->m_pos.x, i->m_pos.y), 5, cv::Scalar(255, 0, 0), -1);
	//}

	//FOR DEBUGGING:
	//int index;
	//int surroundings[6];
	//for (int i = 0; i < m_vlist.size(); i++)
	//{
	//	if (m_vlist[i]->m_tris.size() >= 6)
	//	{
	//		for (int k = 0; k < 6; k++)
	//		{
	//			surroundings[k] = Next_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[k]->m_index);
	//		}
	//	}
	//	else if (m_vlist[i]->m_tris.size() == 5)
	//	{
	//		for (int k = 0; k < 5; k++)
	//		{
	//			surroundings[k] = Next_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[k]->m_index);
	//		}
	//		surroundings[5] = Find_sixth_cloest_vert(m_vlist[i]->m_index);
	//	}
	//	else if (m_vlist[i]->m_tris.size() == 4)
	//	{
	//		for (int k = 0; k < 4; k++)
	//		{
	//			surroundings[k] = Next_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[k]->m_index);
	//		}
	//		int* a;
	//		a = Find_fifth_sixth_cloest_vert(m_vlist[i]->m_index);
	//		surroundings[4] = a[0];
	//		surroundings[5] = a[1];
	//	}
	//	if (surroundings[5] == 3 && (m_vlist[i]->m_pos.x != 0 && m_vlist[i]->m_pos.y != 0 && m_vlist[i]->m_pos.x != m_dim_num - 1 && m_vlist[i]->m_pos.y != m_dim_num - 1))
	//	{
	//		cv::circle(m_Dtexture, cv::Point2i(m_vlist[i]->m_pos.x, m_vlist[i]->m_pos.y), 5, cv::Scalar(0, 0, 255), -1);
	//		for (int j = 0; j < 6; j++)
	//		{
	//			cv::circle(m_Dtexture, cv::Point2i(m_vlist[surroundings[j]]->m_pos.x, m_vlist[surroundings[j]]->m_pos.y), 5, cv::Scalar(0, 155, 255), -1);
	//		}
	//		break;
	//	}
	//}
	//

	//FOR DEBUGGING:
	
	//for (int i = 0; i < m_vlist.size(); i++)
	//{
	//	int surroundings[6];
	//	if (m_vlist[i]->m_tris.size() == 4)
	//	{
	//		surroundings[0] = Next_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[0]->m_index); 
	//		for (int k = 1; k < 5; k++)
	//		{
	//			surroundings[k] = Prev_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[k - 1]->m_index);
	//		}
	//		surroundings[5] = Find_6th_cloest_vert_Boundary(m_vlist[i]->m_index);
	//		
	//	}
	//	else if (m_vlist[i]->m_tris.size() == 3)
	//	{
	//		surroundings[0] = Next_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[0]->m_index);
	//		for (int k = 1; k < 4; k++)
	//		{
	//			surroundings[k] = Prev_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[k - 1]->m_index);
	//		}
	//		int* a;
	//		a = Find_5th_to_6th_cloest_vert_Boundary(m_vlist[i]->m_index);
	//		surroundings[4] = a[0];
	//		surroundings[5] = a[1];
	//		
	//	}
	//	else if (m_vlist[i]->m_tris.size() == 2)
	//	{
	//		//surroundings[0] = Prev_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[0]->m_index);
	//		//for (int k = 1; k < 3; k++)
	//		//{
	//		//	surroundings[k] = Next_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[k]->m_index);
	//		//}
	//		int* a;
	//		a = Find_4th_to_6th_cloest_vert(m_vlist[i]->m_index); // it will return all the surrounding points
	//		surroundings[0] = a[0];
	//		surroundings[1] = a[1];
	//		surroundings[2] = a[2];
	//		surroundings[3] = a[3];
	//		surroundings[4] = a[4];
	//		surroundings[5] = a[5];			
	//
	//		
	//	}
	//	else if (m_vlist[i]->m_tris.size() == 1)
	//	{
	//		//surroundings[0] = Next_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[0]->m_index);
	//		//surroundings[1] = Prev_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[0]->m_index);
	//
	//		int* a;
	//		a = Find_3rd_to_6th_cloest_vert(m_vlist[i]->m_index); // it will return all the surrounding points
	//		surroundings[0] = a[0];
	//		surroundings[1] = a[1];
	//		surroundings[2] = a[2];
	//		surroundings[3] = a[3];
	//		surroundings[4] = a[4];
	//		surroundings[5] = a[5];
	//
	//		if (m_vlist[i]->m_pos.x == 0 || m_vlist[i]->m_pos.x == m_dim_num - 1 || m_vlist[i]->m_pos.y == m_dim_num - 1 || m_vlist[i]->m_pos.y == 0)
	//		{
	//			cv::circle(m_Dtexture, cv::Point2i(m_vlist[i]->m_pos.x, m_vlist[i]->m_pos.y), 5, cv::Scalar(0, 0, 100), -1);
	//
	//			cv::circle(m_Dtexture, cv::Point2i(m_vlist[surroundings[0]]->m_pos.x, m_vlist[surroundings[0]]->m_pos.y), 5, cv::Scalar(0, 0, 255), -1);
	//			cv::circle(m_Dtexture, cv::Point2i(m_vlist[surroundings[1]]->m_pos.x, m_vlist[surroundings[1]]->m_pos.y), 5, cv::Scalar(0, 255, 255), -1);
	//			cv::circle(m_Dtexture, cv::Point2i(m_vlist[surroundings[2]]->m_pos.x, m_vlist[surroundings[2]]->m_pos.y), 5, cv::Scalar(0, 255, 0), -1);
	//			cv::circle(m_Dtexture, cv::Point2i(m_vlist[surroundings[3]]->m_pos.x, m_vlist[surroundings[3]]->m_pos.y), 5, cv::Scalar(255, 255, 0), -1);
	//			cv::circle(m_Dtexture, cv::Point2i(m_vlist[surroundings[4]]->m_pos.x, m_vlist[surroundings[4]]->m_pos.y), 5, cv::Scalar(255, 0, 0), -1);
	//			cv::circle(m_Dtexture, cv::Point2i(m_vlist[surroundings[5]]->m_pos.x, m_vlist[surroundings[5]]->m_pos.y), 5, cv::Scalar(255, 0, 255), -1);
	//			break;
	//		}
	//	}
	//
	//	//double d = 1.0 / pow(sqrt(pow((m_vlist[surroundings[5]]->m_pos.x - m_vlist[i]->m_pos.x), 2) + pow((m_vlist[surroundings[5]]->m_pos.y - m_vlist[i]->m_pos.y), 2)), 2);
	//	//if ((m_vlist[i]->m_pos.x == 0  || m_vlist[i]->m_pos.x == m_dim_num - 1)&& m_vlist[i]->m_pos.y != m_dim_num - 1 && m_vlist[i]->m_pos.y != 0)
	//	//{
	//	//	cv::circle(m_Dtexture, cv::Point2i(m_vlist[i]->m_pos.x, m_vlist[i]->m_pos.y), 5, cv::Scalar(0, 0, 255), -1);
	//	//	for (int j = 0; j < 6; j++)
	//	//	{
	//	//		cv::circle(m_Dtexture, cv::Point2i(m_vlist[surroundings[j]]->m_pos.x, m_vlist[surroundings[j]]->m_pos.y), 5, cv::Scalar(0, 155, 255), -1);
	//	//	}
	//	//	break;
	//	//}
	//	
	//}

	for (auto edge : m_vd->m_elist[0])
	{
		cv::Point2i p1;
		cv::Point2i p2;

		if (edge->IsInfinit())
		{
			if (edge->m_verts[0] == NULL)
			{
				p1 = cv::Point2i(edge->m_infinit_vert.x, edge->m_infinit_vert.y);
				p2 = cv::Point2i(edge->m_verts[1]->m_pos.x, edge->m_verts[1]->m_pos.y);
				cv::line(m_Dtexture, p1, p2, cv::Scalar(0, 0, 0), 2);
			}
			else
			{
				p1 = cv::Point2i(edge->m_verts[0]->m_pos.x, edge->m_verts[0]->m_pos.y);
				p2 = cv::Point2i(edge->m_infinit_vert.x, edge->m_infinit_vert.y);
				cv::line(m_Dtexture, p1, p2, cv::Scalar(0, 0, 0), 2);
			}
		}
		else
		{
			p1 = cv::Point2i(edge->m_verts[0]->m_pos.x, edge->m_verts[0]->m_pos.y);
			p2 = cv::Point2i(edge->m_verts[1]->m_pos.x, edge->m_verts[1]->m_pos.y);
			cv::line(m_Dtexture, p1, p2, cv::Scalar(0, 0, 0), 2);
		}
	}

	//For marking down the statistial points
	//auto it = m_DicHVDEtoTriv.begin();
	//for (int i = 0; i < 5; i ++)
	//{
	//	FDE_Vertex* vert = m_vlist[it->second];
	//	cv::circle(m_Dtexture, cv::Point2i(vert->m_pos.x, vert->m_pos.y), 5, cv::Scalar(0, 0, 255), -1);
	//	cv::putText(m_Dtexture, //target image
	//		std::to_string(it->second), //text
	//		cv::Point2i(vert->m_pos.x, vert->m_pos.y), //top-left position
	//		cv::FONT_HERSHEY_DUPLEX,
	//		1.0,
	//		cv::Scalar(0,0,255), //font color
	//		2);
	//	std::advance(it, 50);
	//}

	//for (auto vertex_index : m_DicHVDEtoTriv)
	//{
	//	FDE_Vertex* vert = m_vlist[vertex_index.second];
	//	if (vert->m_edgeindex != -1)
	//		cv::circle(m_Dtexture, cv::Point2i(vert->m_pos.x, vert->m_pos.y), 5, cv::Scalar(0, 255, 255), -1);
	//	else
	//		cv::circle(m_Dtexture, cv::Point2i(vert->m_pos.x, vert->m_pos.y), 5, cv::Scalar(255, 255, 0), -1);
	//}

	std::ostringstream os;
	os << "./TempMap/" << "DelaunayMesh" << ".png";
	cv::imwrite(os.str(), m_Dtexture);

	std::cout << "Saved image as " << os.str() << "\n";
}
//================================================================================================================
void FDE_Topology::InitializeGrid()
{
	////second BC 4
	//for (auto v : m_vlist)
	//{
	//	int row = v->m_pos.y;
	//	int col = v->m_pos.x;
	//	if (col == m_dim_num - 1 || row == 0)
	//	{
	//		v->m_heat_value = 313.15;
	//		v->m_old_heat_value = 313.15;
	//		v->m_constrainflag = true;
	//	}
	//	else if (col == 0 || row == m_dim_num - 1)
	//	{
	//		v->m_heat_value = 273.15;
	//		v->m_old_heat_value = 273.15;
	//		v->m_constrainflag = false;
	//	}
	//	else
	//	{
	//		v->m_heat_value = 273.15;
	//		v->m_old_heat_value = 273.15;
	//		v->m_constrainflag = false;
	//	}
	//}

	//first BC 4
	for (auto v : m_vlist)
	{
		int row = v->m_pos.y;
		int col = v->m_pos.x;
		if (col == m_dim_num - 1 || row == 0)
		{
			v->m_heat_value = 313.15;
			v->m_old_heat_value = 313.15;
			v->m_constrainflag = true;
		}
		else if (col == 0 || row == m_dim_num - 1)
		{
			v->m_heat_value = 273.15;
			v->m_old_heat_value = 273.15;
			v->m_constrainflag = true;
		}
		else
		{
			v->m_heat_value = 273.15;
			v->m_old_heat_value = 273.15;
			v->m_constrainflag = false;
		}
	}

	//first BC 0
	//for (auto v : m_vlist)
	//{
	//	int row = v->m_pos.y;
	//	int col = v->m_pos.x;
	//	if (row == 0)
	//	{
	//		v->m_heat_value = 313.15;
	//		v->m_old_heat_value = 313.15;
	//		v->m_constrainflag = true;
	//	}
	//	else if (col == m_dim_num - 1 || col == 0 || row == m_dim_num - 1)
	//	{
	//		v->m_heat_value = 273.15;
	//		v->m_old_heat_value = 273.15;
	//		v->m_constrainflag = true;
	//	}
	//	else
	//	{
	//		v->m_heat_value = 273.15;
	//		v->m_old_heat_value = 273.15;
	//		v->m_constrainflag = false;
	//	}
	//}

}

void FDE_Topology::SetHeatCoefficient(std::vector<double>* sites_alpha)
{
	for (int i = 0; i < m_vlist.size(); i++)
	{
		if (i < (*m_sites).size())
		{
			double k = (*sites_alpha)[i] / 255.0;  // calculate the alpha value according the density map's gray scale value
			m_vlist[i]->m_point_alpha = 1.13 * 0.01 * k;
		}
		else
		{
			//for edge point or vertex point
			//m_vlist[i]->m_point_alpha = 1.13 * 0.01 * (0.5 * (m_vlist[m_vlist[i]->m_cell_sites_dic.first]->m_point_alpha + m_vlist[m_vlist[i]->m_cell_sites_dic.second]->m_point_alpha));
			
			// or we can set it with an average value : double alpha = 2.47036*0.0000001;
			m_vlist[i]->m_point_alpha = 1.13 * 0.01 * 0.5;//2.47036 * 0.0000001;
		}
	}
}

void FDE_Topology::SetHeat(int index, double temp)
{
	m_vlist[index]->m_heat_value = temp;
	m_vlist[index]->m_constrainflag = false;
	m_max_temp = std::max(temp, m_max_temp);
	m_min_temp = std::min(temp, m_min_temp);
}

void FDE_Topology::SetHeatBoundary(int index, double temp)
{
	m_vlist[index]->m_heat_value = temp;
	m_vlist[index]->m_constrainflag = true;
	m_max_temp = std::max(temp, m_max_temp);
	m_min_temp = std::min(temp, m_min_temp);
}

double* FDE_Topology::CalculateCoefficient(double dt, double alpha, int coreindex, int* surroundings)
{
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	double weight[6] = { 0.0 };
	double diffx[6] = { 0.0 };
	double diffy[6] = { 0.0 };
	double distance[6] = { 0.0 };
	double maxdistance = 50*0.001;
	for (int k = 0; k < 6; k++)
	{
		diffx[k] = (m_vlist[surroundings[k]]->m_pos.x - m_vlist[coreindex]->m_pos.x) * 0.001;
		diffy[k] = (m_vlist[surroundings[k]]->m_pos.y - m_vlist[coreindex]->m_pos.y) * 0.001;
		distance[k] = sqrt(pow(diffx[k], 2) + pow(diffy[k], 2));
		//maxdistance = std::max(distance[k], maxdistance);
		//weight[k] = 1.0 / pow((diffx[k] * diffx[k] + diffy[k] * diffy[k]),2);

	}
	for (int k = 0; k < 6; k++)
	{
		//double divide = distance[k] / maxdistance;
		//if (distance[k] <= 0.5 * maxdistance)
		//	weight[k] = 2.0 / 3.0 - 4.0 * pow((divide), 2) - 4.0 * pow((divide), 3);
		//else if (distance[k] > 0.5 * maxdistance && distance[k] <= maxdistance)
		//	weight[k] = 4.0 / 3.0 - 4.0 * divide + 4.0 * pow((divide), 2) - 4.0 / 3.0 * pow((divide), 3);
		//else
		//	weight[k] = 0.0;
		weight[k] = 1.0 / pow(sqrt(diffx[k] * diffx[k] + diffy[k] * diffy[k]), 2);//exp(-10 * (diffx[k] * diffx[k] + diffy[k] * diffy[k]));//;
	}
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	double dji[6][5] = { 0.0 };
	for (int j = 0; j < 6; j++)
	{
		double p = 1.0 / pow(weight[j], 2);
		dji[j][0] = diffx[j] / p;
		dji[j][1] = diffy[j] / p;
		dji[j][2] = pow(diffx[j], 2) / (2.0 * p);
		dji[j][3] = pow(diffy[j], 2) / (2.0 * p);
		dji[j][4] = (diffx[j] * diffy[j]) / p;
	}
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	double c[5] = { 0.0 };
	for (int k = 0; k < 5; k++)
	{
		for (int p = 0; p < 6; p++)
		{
			c[k] += dji[p][k];
		}
	}
	double A11 = 0.0, A12 = 0.0, A13 = 0.0, A14 = 0.0, A15 = 0.0,
		A22 = 0.0, A23 = 0.0, A24 = 0.0, A25 = 0.0,
		A33 = 0.0, A34 = 0.0, A35 = 0.0,
		A44 = 0.0, A45 = 0.0,
		A55 = 0.0;
	for (int j = 0; j < 6; j++)
	{
		double p = 1.0 / pow(weight[j], 2);
		A11 += pow(diffx[j], 2) / p;
		A12 += (diffx[j] * diffy[j]) / p;
		A13 += pow(diffx[j], 3) / (2.0 * p);
		A14 += (diffx[j] * pow(diffy[j], 2)) / (2.0 * p);
		A15 += (pow(diffx[j], 2) * diffy[j]) / p;
		A22 += pow(diffy[j], 2) / p;
		A23 += (pow(diffx[j], 2) * diffy[j]) / (2.0 * p);
		A24 += pow(diffy[j], 3) / (2.0 * p);
		A25 += (diffx[j] * pow(diffy[j], 2)) / p;
		A33 += pow(diffx[j], 4) / (4.0 * p);
		A34 += (pow(diffx[j], 2) * pow(diffy[j], 2)) / (4.0 * p);
		A35 += (pow(diffx[j], 3) * diffy[j]) / (2.0 * p);
		A44 += pow(diffy[j], 4) / (4.0 * p);
		A45 += (diffx[j] * pow(diffy[j], 3)) / (2.0 * p);
		A55 += (pow(diffx[j], 2) * pow(diffy[j], 2)) / p;
	}
	Eigen::MatrixXd A_L(5, 5);
	A_L << A11, A12, A13, A14, A15,
		A12, A22, A23, A24, A25,
		A13, A23, A33, A34, A35,
		A14, A24, A34, A44, A45,
		A15, A25, A35, A45, A55;

	Eigen::MatrixXd L(A_L.llt().matrixL());
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	double l[5][5] = { -0.0 };
	for (int j = 0; j < 5; j++)
	{
		for (int k = 0; k < 5; k++)
		{
			l[j][k] = L.coeff(j, k);
		}
	}
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	double M[5][5] = { -0.0 };
	for (int ii = 0; ii < 4; ii++)
	{
		for (int j = 0; j < 4; j++)
		{
			if (j < ii)
			{
				double sum = 0.0;
				for (int k = j; k < (ii - 1); k++)
				{
					sum += l[ii][k] * M[k][j];
				}
				M[ii][j] = (-1.0) * (1.0 / l[ii][ii]) * sum;
			}
			else if (j == ii)
			{
				M[ii][j] = 1.0 / l[ii][ii];
			}
			else
			{
				M[ii][j] = 0;
			}
		}
	}

	//==============================================================================================


	double MLcoeff1 = { 0.0 };
	double MLcoeff2[6] = { 0.0 };

	for (int p = 0; p < 5; p++)
	{
		MLcoeff1 += c[p] * ((M[2][p] / l[2][2]) + (M[3][p] / l[3][3]));
		for (int q = 0; q < 6; q++)
		{
			double e1 = dji[q][p];
			double e2 = (M[2][p] / l[2][2]) + (M[3][p] / l[3][3]);
			MLcoeff2[q] += dji[q][p] * ((M[2][p] / l[2][2]) + (M[3][p] / l[3][3]));
		}
	}

	double efficient[7] = { 0.0 };
	double sum = 0.0; //for testing the sum of weight coefficiences equals to 1
	for (int q = 0; q < 7; q++)
	{
		if (q == 0)
		{
			efficient[q] = 1 + dt * alpha * MLcoeff1;
		}
		else
		{
			efficient[q] = - dt * alpha * MLcoeff2[q - 1];
		}
		sum += efficient[q];
	}

	std::cout << efficient << std::endl;
	return efficient;
}

double* FDE_Topology::CalculateCoefficientBoundaryX(double dt, double alpha, int coreindex, int* surroundings)
{
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	double weight[6] = { 0.0 };
	double diffx[6] = { 0.0 };
	double diffy[6] = { 0.0 };
	double distance[6] = { 0.0 };
	double maxdistance = 50 * 0.001;
	for (int k = 0; k < 6; k++)
	{
		diffx[k] = (m_vlist[surroundings[k]]->m_pos.x - m_vlist[coreindex]->m_pos.x)*0.001;
		diffy[k] = (m_vlist[surroundings[k]]->m_pos.y - m_vlist[coreindex]->m_pos.y)*0.001;
		distance[k] = sqrt(pow(diffx[k], 2) + pow(diffy[k], 2));
		//maxdistance = std::max(distance[k], maxdistance);
		//weight[k] = 1.0 / pow((diffx[k] * diffx[k] + diffy[k] * diffy[k]),2);

	}
	for (int k = 0; k < 6; k++)
	{
		//double divide = distance[k] / maxdistance;
		//if (distance[k] <= 0.5 * maxdistance)
		//	weight[k] = 2.0 / 3.0 - 4.0 * pow((divide), 2) - 4.0 * pow((divide), 3);
		//else if (distance[k] > 0.5 * maxdistance && distance[k] <= maxdistance)
		//	weight[k] = 4.0 / 3.0 - 4.0 * divide + 4.0 * pow((divide), 2) - 4.0 / 3.0 * pow((divide), 3);
		//else
		//	weight[k] = 0.0;
		weight[k] = 1.0 / pow(sqrt(diffx[k] * diffx[k] + diffy[k] * diffy[k]), 2);//exp(-10 * (diffx[k] * diffx[k] + diffy[k] * diffy[k]));//;
	}
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	int sss[6] = { 0.0 };
	memcpy(sss, surroundings, sizeof(sss));

	double dji[6][5] = { 0.0 };
	for (int j = 0; j < 6; j++)
	{
		double p = 1.0 / pow(weight[j], 2);
		dji[j][0] = diffx[j] / p;
		dji[j][1] = diffy[j] / p;
		dji[j][2] = pow(diffx[j], 2) / (2.0 * p);
		dji[j][3] = pow(diffy[j], 2) / (2.0 * p);
		dji[j][4] = (diffx[j] * diffy[j]) / p;
	}
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	double c[5] = { 0.0 };
	for (int k = 0; k < 5; k++)
	{
		for (int p = 0; p < 6; p++)
		{
			c[k] += dji[p][k];
		}
	}
	double A11 = 0.0, A12 = 0.0, A13 = 0.0, A14 = 0.0, A15 = 0.0,
		A22 = 0.0, A23 = 0.0, A24 = 0.0, A25 = 0.0,
		A33 = 0.0, A34 = 0.0, A35 = 0.0,
		A44 = 0.0, A45 = 0.0,
		A55 = 0.0;
	for (int j = 0; j < 6; j++)
	{
		double p = 1.0 / pow(weight[j], 2);
		A11 += pow(diffx[j], 2) / p;
		A12 += (diffx[j] * diffy[j]) / p;
		A13 += pow(diffx[j], 3) / (2.0 * p);
		A14 += (diffx[j] * pow(diffy[j], 2)) / (2.0 * p);
		A15 += (pow(diffx[j], 2) * diffy[j]) / p;
		A22 += pow(diffy[j], 2) / p;
		A23 += (pow(diffx[j], 2) * diffy[j]) / (2.0 * p);
		A24 += pow(diffy[j], 3) / (2.0 * p);
		A25 += (diffx[j] * pow(diffy[j], 2)) / p;
		A33 += pow(diffx[j], 4) / (4.0 * p);
		A34 += (pow(diffx[j], 2) * pow(diffy[j], 2)) / (4.0 * p);
		A35 += (pow(diffx[j], 3) * diffy[j]) / (2.0 * p);
		A44 += pow(diffy[j], 4) / (4.0 * p);
		A45 += (diffx[j] * pow(diffy[j], 3)) / (2.0 * p);
		A55 += (pow(diffx[j], 2) * pow(diffy[j], 2)) / p;
	}
	Eigen::MatrixXd A_L(5, 5);
	A_L << A11, A12, A13, A14, A15,
		A12, A22, A23, A24, A25,
		A13, A23, A33, A34, A35,
		A14, A24, A34, A44, A45,
		A15, A25, A35, A45, A55;

	Eigen::MatrixXd L(A_L.llt().matrixL());
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	double l[5][5] = { 0.0 };
	for (int j = 0; j < 5; j++)
	{
		for (int k = 0; k < 5; k++)
		{
			l[j][k] = L.coeff(j, k);
		}
	}
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	double M[5][5] = { -0.0 };
	for (int ii = 0; ii < 4; ii++)
	{
		for (int j = 0; j < 4; j++)
		{
			if (j < ii)
			{
				double sum = 0.0;
				for (int k = j; k < (ii - 1); k++)
				{
					sum += l[ii][k] * M[k][j];
				}
				M[ii][j] = (-1.0) * (1.0 / l[ii][ii]) * sum;
			}
			else if (j == ii)
			{
				M[ii][j] = 1.0 / l[ii][ii];
			}
			else
			{
				M[ii][j] = 0;
			}
		}
	}

	//==========================================Changed place======================================

	double MLcoeff1 = { 0.0 };
	double MLcoeff2[6] = { 0.0 };

	for (int p = 0; p < 5; p++)
	{
		MLcoeff1 += c[p] * (M[0][p] / l[0][0]);
		for (int q = 0; q < 6; q++)
		{
			double e1 = dji[q][p];
			double e2 = (M[0][p] / l[0][0] );
			MLcoeff2[q] += e1*e2;
		}
	}

	double efficient[7] = { 0.0 };
	double sum = 0.0; //for testing the sum of weight coefficiences equals to 1
	for (int q = 0; q < 7; q++)
	{
		if (q == 0)
		{
			efficient[q] = - MLcoeff1;
		}
		else
		{
			efficient[q] =  MLcoeff2[q - 1];
		}
		sum += efficient[q];
	}

	return efficient;
}

double* FDE_Topology::CalculateCoefficientBoundaryY(double dt, double alpha, int coreindex, int* surroundings)
{
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	double weight[6] = { 0.0 };
	double diffx[6] = { 0.0 };
	double diffy[6] = { 0.0 };
	double distance[6] = { 0.0 };
	double maxdistance = 50 * 0.001;
	for (int k = 0; k < 6; k++)
	{
		diffx[k] = (m_vlist[surroundings[k]]->m_pos.x - m_vlist[coreindex]->m_pos.x)*0.001;
		diffy[k] = (m_vlist[surroundings[k]]->m_pos.y - m_vlist[coreindex]->m_pos.y)*0.001;
		distance[k] = sqrt(pow(diffx[k], 2) + pow(diffy[k], 2));
		//maxdistance = std::max(distance[k], maxdistance);
		//weight[k] = 1.0 / pow((diffx[k] * diffx[k] + diffy[k] * diffy[k]),2);

	}
	for (int k = 0; k < 6; k++)
	{
		//double divide = distance[k] / maxdistance;
		//if (distance[k] <= 0.5 * maxdistance)
		//	weight[k] = 2.0 / 3.0 - 4.0 * pow((divide), 2) - 4.0 * pow((divide), 3);
		//else if (distance[k] > 0.5 * maxdistance && distance[k] <= maxdistance)
		//	weight[k] = 4.0 / 3.0 - 4.0 * divide + 4.0 * pow((divide), 2) - 4.0 / 3.0 * pow((divide), 3);
		//else
		//	weight[k] = 0.0;
		weight[k] = 1.0 / pow(sqrt(diffx[k] * diffx[k] + diffy[k] * diffy[k]), 2);//exp(-10 * (diffx[k] * diffx[k] + diffy[k] * diffy[k]));//;
	}
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	double dji[6][5] = { 0.0 };
	int sss[6] = { 0.0 };
	memcpy(sss, surroundings, sizeof(sss));
	
	for (int j = 0; j < 6; j++)
	{
		double p = 1.0 / pow(weight[j], 2);
		dji[j][0] = diffx[j] / p;
		dji[j][1] = diffy[j] / p;
		dji[j][2] = pow(diffx[j], 2) / (2.0 * p);
		dji[j][3] = pow(diffy[j], 2) / (2.0 * p);
		dji[j][4] = (diffx[j] * diffy[j]) / p;
	}
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	double c[5] = { 0.0 };
	for (int k = 0; k < 5; k++)
	{
		for (int p = 0; p < 6; p++)
		{
			c[k] += dji[p][k];
		}
	}
	double A11 = 0.0, A12 = 0.0, A13 = 0.0, A14 = 0.0, A15 = 0.0,
		A22 = 0.0, A23 = 0.0, A24 = 0.0, A25 = 0.0,
		A33 = 0.0, A34 = 0.0, A35 = 0.0,
		A44 = 0.0, A45 = 0.0,
		A55 = 0.0;
	for (int j = 0; j < 6; j++)
	{
		double p = 1.0 / pow(weight[j], 2);
		A11 += pow(diffx[j], 2) / p;
		A12 += (diffx[j] * diffy[j]) / p;
		A13 += pow(diffx[j], 3) / (2.0 * p);
		A14 += (diffx[j] * pow(diffy[j], 2)) / (2.0 * p);
		A15 += (pow(diffx[j], 2) * diffy[j]) / p;
		A22 += pow(diffy[j], 2) / p;
		A23 += (pow(diffx[j], 2) * diffy[j]) / (2.0 * p);
		A24 += pow(diffy[j], 3) / (2.0 * p);
		A25 += (diffx[j] * pow(diffy[j], 2)) / p;
		A33 += pow(diffx[j], 4) / (4.0 * p);
		A34 += (pow(diffx[j], 2) * pow(diffy[j], 2)) / (4.0 * p);
		A35 += (pow(diffx[j], 3) * diffy[j]) / (2.0 * p);
		A44 += pow(diffy[j], 4) / (4.0 * p);
		A45 += (diffx[j] * pow(diffy[j], 3)) / (2.0 * p);
		A55 += (pow(diffx[j], 2) * pow(diffy[j], 2)) / p;
	}
	Eigen::MatrixXd A_L(5, 5);
	A_L << A11, A12, A13, A14, A15,
		A12, A22, A23, A24, A25,
		A13, A23, A33, A34, A35,
		A14, A24, A34, A44, A45,
		A15, A25, A35, A45, A55;

	Eigen::MatrixXd L(A_L.llt().matrixL());
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	double l[5][5] = { 0.0 };
	for (int j = 0; j < 5; j++)
	{
		for (int k = 0; k < 5; k++)
		{
			l[j][k] = L.coeff(j, k);
		}
	}
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	double M[5][5] = { -0.0 };
	for (int ii = 0; ii < 4; ii++)
	{
		for (int j = 0; j < 4; j++)
		{
			if (j < ii)
			{
				double sum = 0.0;
				for (int k = j; k < (ii - 1); k++)
				{
					sum += l[ii][k] * M[k][j];
				}
				M[ii][j] = (-1.0) * (1.0 / l[ii][ii]) * sum;
			}
			else if (j == ii)
			{
				M[ii][j] = 1.0 / l[ii][ii];
			}
			else
			{
				M[ii][j] = 0;
			}
		}
	}

	//==========================================Changed place======================================

	double MLcoeff1 = { 0.0 };
	double MLcoeff2[6] = { 0.0 };

	for (int p = 0; p < 5; p++)
	{
		MLcoeff1 += c[p] * (M[1][p] / l[1][1]);
		for (int q = 0; q < 6; q++)
		{
			double e1 = dji[q][p];
			double e2 = (M[1][p] / l[1][1]);
			MLcoeff2[q] += e1 * e2;
		}
	}

	double efficient[7] = { 0.0 };
	double sum = 0.0; //for testing the sum of weight coefficiences equals to 1
	for (int q = 0; q < 7; q++)
	{
		if (q == 0)
		{
			efficient[q] = -MLcoeff1;
		}
		else
		{
			efficient[q] = MLcoeff2[q - 1];
		}
		sum += efficient[q];
	}

	return efficient;
}

void FDE_Topology::UpdateHeat_implicit(double step_length)
{
	double heat_flux_C = -1;

	double dt = step_length;//50
	int m = m_vlist.size();

	//1. Assembly:
	std::vector<T> tripletList;   // list of non-zeros coefficients the triple value
	Eigen::VectorXd b(m);     // b vector at the right side of the equation

	//2. BuildProblem: push back coefficient
	b.setZero();

	for (int i = 0; i < m_vlist.size(); i++)
	{
		int row = m_vlist[i]->m_pos.y;
		int col = m_vlist[i]->m_pos.x;
		
		double alpha = m_vlist[i]->m_point_alpha;
		//b[i] = m_vlist[i]->m_heat_value;

		if (row == 0 || col == 0 || row == m_dim_num - 1 || col == m_dim_num - 1)
		{
			
			//0. puretop //1. pureleft //2. pureright //3. topleft //4. topright
			//tripletList.push_back(T(i, i, 1));
			if (m_boundary_status == 0)
			{
				if (row == 0 && col != 0 && col != m_dim_num - 1)
				{
					tripletList.push_back(T(i, i, 1));
					b[i] = m_vlist[i]->m_heat_value;
				}
				else  //first bc
				{
					tripletList.push_back(T(i, i, 1));
					b[i] = m_vlist[i]->m_heat_value;
				}
				//else //second bc
				//{
				//	b[i] = heat_flux_C;
				//	int surroundings[6];
				//	if (m_vlist[i]->m_tris.size() == 4)
				//	{
				//		surroundings[0] = Prev_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[0]->m_index);
				//		for (int k = 1; k < 5; k++)
				//		{
				//			surroundings[k] = Next_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[k-1]->m_index);
				//		}
				//		surroundings[5] = Find_6th_cloest_vert_Boundary(m_vlist[i]->m_index);
				//	}
				//	else if (m_vlist[i]->m_tris.size() == 3)
				//	{
				//		surroundings[0] = Prev_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[0]->m_index);
				//		for (int k = 1; k < 4; k++)
				//		{
				//			surroundings[k] = Next_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[k-1]->m_index);
				//		}
				//		int* a;
				//		a = Find_5th_to_6th_cloest_vert_Boundary(m_vlist[i]->m_index);
				//		surroundings[4] = a[0];
				//		surroundings[5] = a[1];
				//	}
				//	else if (m_vlist[i]->m_tris.size() == 2)
				//	{
				//		int* a;
				//		a = Find_4th_to_6th_cloest_vert(m_vlist[i]->m_index); // it will return all the surrounding points
				//		surroundings[0] = a[0];
				//		surroundings[1] = a[1];
				//		surroundings[2] = a[2];
				//		surroundings[3] = a[3];
				//		surroundings[4] = a[4];
				//		surroundings[5] = a[5];
				//	}
				//	else if (m_vlist[i]->m_tris.size() == 1)
				//	{
				//		int* a;
				//		a = Find_3rd_to_6th_cloest_vert(m_vlist[i]->m_index); // it will return all the surrounding points
				//		surroundings[0] = a[0];
				//		surroundings[1] = a[1];
				//		surroundings[2] = a[2];
				//		surroundings[3] = a[3];
				//		surroundings[4] = a[4];
				//		surroundings[5] = a[5];
				//	}
				//
				//	double efficient[7];
				//	if ((col == 0 && row != m_dim_num - 1) || (col == m_dim_num - 1 && row != m_dim_num - 1))
				//		memcpy(efficient, CalculateCoefficientBoundaryX(dt, alpha, i, surroundings), sizeof(efficient));
				//	else
				//		memcpy(efficient, CalculateCoefficientBoundaryY(dt, alpha, i, surroundings), sizeof(efficient));
				//	
				//	tripletList.push_back(T(i, i, efficient[0]));
				//	tripletList.push_back(T(i, surroundings[0], efficient[1]));
				//	tripletList.push_back(T(i, surroundings[1], efficient[2]));
				//	tripletList.push_back(T(i, surroundings[2], efficient[3]));
				//	tripletList.push_back(T(i, surroundings[3], efficient[4]));
				//	tripletList.push_back(T(i, surroundings[4], efficient[5]));
				//	tripletList.push_back(T(i, surroundings[5], efficient[6]));
				//}
			}
			////1. pureleft
			else if (m_boundary_status == 1)
			{
				if (col == 0 && row != 0 && row != m_dim_num - 1)
				{
					tripletList.push_back(T(i, i, 1));
					b[i] = m_vlist[i]->m_heat_value;
				}
				else  //first bc
				{
					tripletList.push_back(T(i, i, 1));
					b[i] = m_vlist[i]->m_heat_value;
				}
				//else
				//{
				//	b[i] = heat_flux_C;
				//	int surroundings[6] = { -1,-1,-1,-1,-1,-1 };
				//	if (m_vlist[i]->m_tris.size() == 4)
				//	{
				//		surroundings[0] = Prev_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[0]->m_index);
				//		for (int k = 1; k < 5; k++)
				//		{
				//			surroundings[k] = Next_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[k - 1]->m_index);
				//		}
				//		surroundings[5] = Find_6th_cloest_vert_Boundary(m_vlist[i]->m_index);
				//	}
				//	else if (m_vlist[i]->m_tris.size() == 3)
				//	{
				//		surroundings[0] = Prev_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[0]->m_index);
				//		for (int k = 1; k < 4; k++)
				//		{
				//			surroundings[k] = Next_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[k - 1]->m_index);
				//		}
				//		int* a;
				//		a = Find_5th_to_6th_cloest_vert_Boundary(m_vlist[i]->m_index);
				//		surroundings[4] = a[0];
				//		surroundings[5] = a[1];
				//	}
				//	else if (m_vlist[i]->m_tris.size() == 2)
				//	{
				//		int* a;
				//		a = Find_4th_to_6th_cloest_vert(m_vlist[i]->m_index); // it will return all the surrounding points
				//		surroundings[0] = a[0];
				//		surroundings[1] = a[1];
				//		surroundings[2] = a[2];
				//		surroundings[3] = a[3];
				//		surroundings[4] = a[4];
				//		surroundings[5] = a[5];
				//	}
				//	else if (m_vlist[i]->m_tris.size() == 1)
				//	{
				//		int* a;
				//		a = Find_3rd_to_6th_cloest_vert(m_vlist[i]->m_index); // it will return all the surrounding points
				//		surroundings[0] = a[0];
				//		surroundings[1] = a[1];
				//		surroundings[2] = a[2];
				//		surroundings[3] = a[3];
				//		surroundings[4] = a[4];
				//		surroundings[5] = a[5];
				//	}
				//
				//	double efficient[7];
				//	if ((row == 0 && col != m_dim_num - 1) || (row == m_dim_num - 1 && col != m_dim_num - 1))
				//		memcpy(efficient, CalculateCoefficientBoundaryY(dt, alpha, i, surroundings), sizeof(efficient));
				//	else
				//		memcpy(efficient, CalculateCoefficientBoundaryX(dt, alpha, i, surroundings), sizeof(efficient));
				//
				//	tripletList.push_back(T(i, i, efficient[0]));
				//	tripletList.push_back(T(i, surroundings[0], efficient[1]));
				//	tripletList.push_back(T(i, surroundings[1], efficient[2]));
				//	tripletList.push_back(T(i, surroundings[2], efficient[3]));
				//	tripletList.push_back(T(i, surroundings[3], efficient[4]));
				//	tripletList.push_back(T(i, surroundings[4], efficient[5]));
				//	tripletList.push_back(T(i, surroundings[5], efficient[6]));
				//}
			}
			//2. pureright
			else if (m_boundary_status == 2)
			{
				if (col == m_dim_num - 1 && row != 0 && row != m_dim_num - 1)
				{
					tripletList.push_back(T(i, i, 1));
					b[i] = m_vlist[i]->m_heat_value;
				}
				else  //first bc
				{
					tripletList.push_back(T(i, i, 1));
					b[i] = m_vlist[i]->m_heat_value;
				}
				//else
				//{
				//	b[i] = heat_flux_C;
				//	int surroundings[6] = { -1,-1,-1,-1,-1,-1 };
				//	if (m_vlist[i]->m_tris.size() == 4)
				//	{
				//		surroundings[0] = Prev_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[0]->m_index);
				//		for (int k = 1; k < 5; k++)
				//		{
				//			surroundings[k] = Next_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[k - 1]->m_index);
				//		}
				//		surroundings[5] = Find_6th_cloest_vert_Boundary(m_vlist[i]->m_index);
				//	}
				//	else if (m_vlist[i]->m_tris.size() == 3)
				//	{
				//		surroundings[0] = Prev_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[0]->m_index);
				//		for (int k = 1; k < 4; k++)
				//		{
				//			surroundings[k] = Next_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[k - 1]->m_index);
				//		}
				//		int* a;
				//		a = Find_5th_to_6th_cloest_vert_Boundary(m_vlist[i]->m_index);
				//		surroundings[4] = a[0];
				//		surroundings[5] = a[1];
				//	}
				//	else if (m_vlist[i]->m_tris.size() == 2)
				//	{
				//		int* a;
				//		a = Find_4th_to_6th_cloest_vert(m_vlist[i]->m_index); // it will return all the surrounding points
				//		surroundings[0] = a[0];
				//		surroundings[1] = a[1];
				//		surroundings[2] = a[2];
				//		surroundings[3] = a[3];
				//		surroundings[4] = a[4];
				//		surroundings[5] = a[5];
				//	}
				//	else if (m_vlist[i]->m_tris.size() == 1)
				//	{
				//		int* a;
				//		a = Find_3rd_to_6th_cloest_vert(m_vlist[i]->m_index); // it will return all the surrounding points
				//		surroundings[0] = a[0];
				//		surroundings[1] = a[1];
				//		surroundings[2] = a[2];
				//		surroundings[3] = a[3];
				//		surroundings[4] = a[4];
				//		surroundings[5] = a[5];
				//	}
				//
				//	double efficient[7];
				//	if ((row == 0 && col != 0) || (row == m_dim_num - 1 && col != 0))
				//		memcpy(efficient, CalculateCoefficientBoundaryY(dt, alpha, i, surroundings), sizeof(efficient));
				//	else
				//		memcpy(efficient, CalculateCoefficientBoundaryX(dt, alpha, i, surroundings), sizeof(efficient));
				//
				//	tripletList.push_back(T(i, i, efficient[0]));
				//	tripletList.push_back(T(i, surroundings[0], efficient[1]));
				//	tripletList.push_back(T(i, surroundings[1], efficient[2]));
				//	tripletList.push_back(T(i, surroundings[2], efficient[3]));
				//	tripletList.push_back(T(i, surroundings[3], efficient[4]));
				//	tripletList.push_back(T(i, surroundings[4], efficient[5]));
				//	tripletList.push_back(T(i, surroundings[5], efficient[6]));
				//}
			}
			//3. topleft
			else if (m_boundary_status == 3)
			{
				if (row == 0 && col != m_dim_num - 1)
				{
					tripletList.push_back(T(i, i, 1));
					b[i] = m_vlist[i]->m_heat_value;
				}
				else if (col == 0 && row != m_dim_num - 1)
				{
					tripletList.push_back(T(i, i, 1));
					b[i] = m_vlist[i]->m_heat_value;
				}
				else  //first bc
				{
					tripletList.push_back(T(i, i, 1));
					b[i] = m_vlist[i]->m_heat_value;
				}
				//else
				//{
				//	b[i] = heat_flux_C;
				//	int surroundings[6] = { -1,-1,-1,-1,-1,-1 };
				//	if (m_vlist[i]->m_tris.size() == 4)
				//	{
				//		surroundings[0] = Prev_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[0]->m_index);
				//		for (int k = 1; k < 5; k++)
				//		{
				//			surroundings[k] = Next_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[k - 1]->m_index);
				//		}
				//		surroundings[5] = Find_6th_cloest_vert_Boundary(m_vlist[i]->m_index);
				//	}
				//	else if (m_vlist[i]->m_tris.size() == 3)
				//	{
				//		surroundings[0] = Prev_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[0]->m_index);
				//		for (int k = 1; k < 4; k++)
				//		{
				//			surroundings[k] = Next_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[k - 1]->m_index);
				//		}
				//		int* a;
				//		a = Find_5th_to_6th_cloest_vert_Boundary(m_vlist[i]->m_index);
				//		surroundings[4] = a[0];
				//		surroundings[5] = a[1];
				//	}
				//	else if (m_vlist[i]->m_tris.size() == 2)
				//	{
				//		int* a;
				//		a = Find_4th_to_6th_cloest_vert(m_vlist[i]->m_index); // it will return all the surrounding points
				//		surroundings[0] = a[0];
				//		surroundings[1] = a[1];
				//		surroundings[2] = a[2];
				//		surroundings[3] = a[3];
				//		surroundings[4] = a[4];
				//		surroundings[5] = a[5];
				//	}
				//	else if (m_vlist[i]->m_tris.size() == 1)
				//	{
				//		int* a;
				//		a = Find_3rd_to_6th_cloest_vert(m_vlist[i]->m_index); // it will return all the surrounding points
				//		surroundings[0] = a[0];
				//		surroundings[1] = a[1];
				//		surroundings[2] = a[2];
				//		surroundings[3] = a[3];
				//		surroundings[4] = a[4];
				//		surroundings[5] = a[5];
				//	}
				//
				//	double efficient[7];
				//	if ((col == m_dim_num - 1 && row != m_dim_num - 1))
				//		memcpy(efficient, CalculateCoefficientBoundaryX(dt, alpha, i, surroundings), sizeof(efficient));
				//	else
				//		memcpy(efficient, CalculateCoefficientBoundaryY(dt, alpha, i, surroundings), sizeof(efficient));
				//
				//	tripletList.push_back(T(i, i, efficient[0]));
				//	tripletList.push_back(T(i, surroundings[0], efficient[1]));
				//	tripletList.push_back(T(i, surroundings[1], efficient[2]));
				//	tripletList.push_back(T(i, surroundings[2], efficient[3]));
				//	tripletList.push_back(T(i, surroundings[3], efficient[4]));
				//	tripletList.push_back(T(i, surroundings[4], efficient[5]));
				//	tripletList.push_back(T(i, surroundings[5], efficient[6]));
				//
				//}
			}
			//4. topright
			else if (m_boundary_status == 4)
			{
				if (row == 0 && col != 0)
				{
					tripletList.push_back(T(i, i, 1));
					b[i] = m_vlist[i]->m_heat_value;
				}
				else if (col == m_dim_num - 1 && row != m_dim_num - 1)
				{
					tripletList.push_back(T(i, i, 1));
					b[i] = m_vlist[i]->m_heat_value;
				}
				//first BC
				else
				{
					tripletList.push_back(T(i, i, 1));
					b[i] = m_vlist[i]->m_heat_value;
				}
				//second BC
				//else 
				//{	
				//	b[i] = heat_flux_C;
				//	int surroundings[6]= {-1,-1,-1,-1,-1,-1};
				//	if (m_vlist[i]->m_tris.size() == 4)
				//	{
				//		surroundings[0] = Prev_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[0]->m_index);
				//		for (int k = 1; k < 5; k++)
				//		{
				//			surroundings[k] = Next_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[k-1]->m_index);
				//		}
				//		surroundings[5] = Find_6th_cloest_vert_Boundary(m_vlist[i]->m_index);
				//	}
				//	else if (m_vlist[i]->m_tris.size() == 3)
				//	{
				//		surroundings[0] = Prev_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[0]->m_index);
				//		for (int k = 1; k < 4; k++)
				//		{
				//			surroundings[k] = Next_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[k-1]->m_index);
				//		}
				//		int* a;
				//		a = Find_5th_to_6th_cloest_vert_Boundary(m_vlist[i]->m_index);
				//		surroundings[4] = a[0];
				//		surroundings[5] = a[1];
				//	}
				//	else if (m_vlist[i]->m_tris.size() == 2)
				//	{
				//		//surroundings[0] = Prev_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[0]->m_index);
				//		//for (int k = 1; k < 3; k++)
				//		//{
				//		//	surroundings[k] = Next_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[k]->m_index);
				//		//}
				//		int* a;
				//		a = Find_4th_to_6th_cloest_vert(m_vlist[i]->m_index); // it will return all the surrounding points
				//		surroundings[0] = a[0];
				//		surroundings[1] = a[1];
				//		surroundings[2] = a[2];
				//		surroundings[3] = a[3];
				//		surroundings[4] = a[4];
				//		surroundings[5] = a[5];
				//	}
				//	else if (m_vlist[i]->m_tris.size() == 1)
				//	{
				//		//surroundings[0] = Next_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[0]->m_index);
				//		//surroundings[1] = Prev_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[0]->m_index);
				//
				//		int* a;
				//		a = Find_3rd_to_6th_cloest_vert(m_vlist[i]->m_index); // it will return all the surrounding points
				//		surroundings[0] = a[0];
				//		surroundings[1] = a[1];
				//		surroundings[2] = a[2];
				//		surroundings[3] = a[3];
				//		surroundings[4] = a[4];
				//		surroundings[5] = a[5];
				//	}
				//
				//	double efficient[7];
				//	if ((col == 0 && row != m_dim_num - 1))
				//		memcpy(efficient, CalculateCoefficientBoundaryX(dt, alpha, i, surroundings), sizeof(efficient));
				//	else
				//		memcpy(efficient, CalculateCoefficientBoundaryY(dt, alpha, i, surroundings), sizeof(efficient));
				//
				//	tripletList.push_back(T(i, i, efficient[0]));
				//	tripletList.push_back(T(i, surroundings[0], efficient[1]));
				//	tripletList.push_back(T(i, surroundings[1], efficient[2]));
				//	tripletList.push_back(T(i, surroundings[2], efficient[3]));
				//	tripletList.push_back(T(i, surroundings[3], efficient[4]));
				//	tripletList.push_back(T(i, surroundings[4], efficient[5]));
				//	tripletList.push_back(T(i, surroundings[5], efficient[6]));
				//	
				//}

			}
		}
		else
		{
			b[i] = m_vlist[i]->m_heat_value;
			//======================================Matrix Calculation======================================
			int surroundings[6] = {-1,-1,-1,-1,-1,-1};
			if (m_vlist[i]->m_tris.size() >= 6)
			{
				for (int k = 0; k < 6; k++)
				{
					surroundings[k] = Next_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[k]->m_index);
				}
			}
			else if (m_vlist[i]->m_tris.size() == 5)
			{
				for (int k = 0; k < 5; k++)
				{
					surroundings[k] = Next_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[k]->m_index);
				}
				surroundings[5] = Find_6th_cloest_vert(m_vlist[i]->m_index);
			}
			else if (m_vlist[i]->m_tris.size() == 4)
			{
				for (int k = 0; k < 4; k++)
				{
					surroundings[k] = Next_Vert(m_vlist[i]->m_index, m_vlist[i]->m_tris[k]->m_index);
				}
				int* a;
				a = Find_5th_to_6th_cloest_vert(m_vlist[i]->m_index);
				surroundings[4] = a[0];
				surroundings[5] = a[1];
			}
			
			double efficient[7];
			memcpy(efficient, CalculateCoefficient(dt, alpha, i, surroundings),sizeof(efficient));

			tripletList.push_back(T(i, i, efficient[0]));
			tripletList.push_back(T(i, surroundings[0], efficient[1]));
			tripletList.push_back(T(i, surroundings[1], efficient[2]));
			tripletList.push_back(T(i, surroundings[2], efficient[3]));
			tripletList.push_back(T(i, surroundings[3], efficient[4]));
			tripletList.push_back(T(i, surroundings[4], efficient[5]));
			tripletList.push_back(T(i, surroundings[5], efficient[6]));
		}
	}

	//3. Construct A
	SpMat A(m, m);            // A matrix at the left side of the equation
	A.setFromTriplets(tripletList.begin(), tripletList.end());  // sent the triple coefficient to matrix
	A.makeCompressed();
	//std::cout << A;

	//4. Solve problem:
	Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver(A);
	//Eigen::SimplicialCholesky<SpMat> solver(A);
	solver.compute(A);
	if (solver.info() != Eigen::Success)
	{
		qDebug() << "Compute Matrix is error";
		return;
	}
	Eigen::VectorXd x = solver.solve(b);

	//5. update the solved value
	for (int i = 0; i < m; i++)
	{
		if (m_vlist[i]->m_constrainflag != true)
		{
			m_vlist[i]->m_old_heat_value = m_vlist[i]->m_heat_value;
			m_vlist[i]->m_heat_value = (x[i]);

		}
		//if (m_vlist[i].ReturnDamageHoleflag() && m_Points[i].ReturnTemp() < 25)
		//	m_Points[i].SetTemp(m_Points[i].ReturnTemp() + 2);
		m_max_temp = std::max(m_vlist[i]->m_heat_value, m_max_temp);
		m_min_temp = std::min(m_vlist[i]->m_heat_value, m_min_temp);
	}
	std::cout << m_max_temp << "," << m_min_temp << std::endl;

	//6. update stress of edge point
	UpdateEdgeStress();

	//7. record stress information
	//OutPutStressExcel();
}

void FDE_Topology::UpdateEdgeStress()
{
	cv::Mat test = cv::Mat::zeros(m_dim_num, m_dim_num, CV_8UC3); //color_CV_32FC3
	test.setTo(255);

	m_max_normal_force = -1;
	m_min_normal_force = 100000;

	for (auto hashedge : m_DicHVDEtoTriv)
	{
		std::vector<int> nuclvert_index;
		FDE_Vertex* edgepoint = m_vlist[hashedge.second];
		for (auto neighbor_corners : edgepoint->m_corners) //traverse all the edge
		{
			if (neighbor_corners->m_prev->m_oppo == NULL)
			{
				FDE_Edge* neiboredge = neighbor_corners->m_prev->m_edge;
				if (neiboredge->m_verts[0]->m_edgeindex == -1 && neiboredge->m_verts[0]->m_isonVert == false)
				{
					nuclvert_index.push_back(neiboredge->m_verts[0]->m_index);
				}
				else if (neiboredge->m_verts[1]->m_edgeindex == -1 && neiboredge->m_verts[1]->m_isonVert == false)
				{
					nuclvert_index.push_back(neiboredge->m_verts[1]->m_index);
				}
	
				neiboredge = neighbor_corners->m_next->m_edge;
				if (neiboredge->m_verts[0]->m_edgeindex == -1 && neiboredge->m_verts[0]->m_isonVert == false)
				{
					nuclvert_index.push_back(neiboredge->m_verts[0]->m_index);
				}
				else if (neiboredge->m_verts[1]->m_edgeindex == -1 && neiboredge->m_verts[1]->m_isonVert == false)
				{
					nuclvert_index.push_back(neiboredge->m_verts[1]->m_index);
				}
			}
			else
			{
				FDE_Edge* neiboredge = neighbor_corners->m_next->m_edge;
				if (neiboredge->m_verts[0]->m_edgeindex == -1 && neiboredge->m_verts[0]->m_isonVert == false)
				{
					nuclvert_index.push_back(neiboredge->m_verts[0]->m_index);
				}
				else if (neiboredge->m_verts[1]->m_edgeindex == -1 && neiboredge->m_verts[1]->m_isonVert == false)
				{
					nuclvert_index.push_back(neiboredge->m_verts[1]->m_index);
				}
			}
		}
		CalculateStress(nuclvert_index, hashedge.second);
	}

	std::cout << "m_max_normal_force = " << m_max_normal_force << std::endl;
	std::cout << "m_min_normal_force = " << m_min_normal_force << std::endl;

	cv::Mat out = cv::Mat::zeros(m_dim_num, m_dim_num, CV_8UC3); //color_heat
	out.setTo(255);
	for (auto hashedge : m_DicHVDEtoTriv)
	{
		FDE_Vertex* EV = m_vlist[hashedge.second];
		double n = m_max_normal_force - m_min_normal_force;
		QVector3D color = 255*color_t_Heat((cv::norm(EV->m_force.first) - m_min_normal_force) / n);//log10(glm::length(EV->m_force.first))/log10(m_max_normal_force)
		
		HVD::edge* E = m_vd->m_elist[0][hashedge.first];
		cv::Point2i p1, p2;
		if (E->IsInfinit())
		{
			if (E->m_verts[0] == NULL)
			{
				p1 = cv::Point2i(E->m_infinit_vert.x, E->m_infinit_vert.y);
				p2 = cv::Point2i(E->m_verts[1]->m_pos.x, E->m_verts[1]->m_pos.y);
	
				cv::line(out, p1, p2, cv::Scalar(color.z(), color.y(), color.x()), 2);
			}
			else
			{
				p1 = cv::Point2i(E->m_verts[0]->m_pos.x, E->m_verts[0]->m_pos.y);
				p2 = cv::Point2i(E->m_infinit_vert.x, E->m_infinit_vert.y);
	
				cv::line(out, p1, p2, cv::Scalar(color.z(), color.y(), color.x()), 2);
			}
		}
		else
		{
			p1 = cv::Point2i(E->m_verts[0]->m_pos.x, E->m_verts[0]->m_pos.y);
			p2 = cv::Point2i(E->m_verts[1]->m_pos.x, E->m_verts[1]->m_pos.y);
	
			cv::line(out, p1, p2, cv::Scalar(color.z(), color.y(), color.x()), 2);
		}
	}
	cv::imwrite("./TempMap/test_force.png", out);
}

double uniformlengthh(double l)
{
	return l / 1000.0 * 5.0;
}

void FDE_Topology::CalculateStress(std::vector<int> nuclvert_vindex, int edgepoint_vindex)
{
	FDE_Vertex* N1 = m_vlist[nuclvert_vindex[0]];
	FDE_Vertex* N2 = m_vlist[nuclvert_vindex[1]];
	FDE_Vertex* EV = m_vlist[edgepoint_vindex];
	double delta_T_N1 = N1->m_heat_value - N1->m_old_heat_value;
	double delta_T_N2 = N2->m_heat_value - N2->m_old_heat_value;
	double delta_T_EV = EV->m_heat_value - EV->m_old_heat_value;

	double YongsModule = 30.0 * 1000000000; //concret one (GPa)
	double saperatenumber = 5;  //saperate number of the bar, will insert same number temp points
	double ThermalExpansionC = 12 * 0.000001; //concrete
	double Area = 3.1415 * pow(0.5/2.0, 2);

	//truss one 
	cv::Point2d truss1 = cv::Point2d(N1->m_pos.x, N1->m_pos.y) - cv::Point2d(EV->m_pos.x, EV->m_pos.y);
	double truss1_len = uniformlengthh(cv::norm(truss1));
	double delta_tr1_len = truss1_len / saperatenumber;
	double expansion1 = 0; //-1 shrink, 1 compress
	double dT_old1 = (N1->m_old_heat_value - EV->m_old_heat_value) / saperatenumber;
	double dT_new1 = (N1->m_heat_value - EV->m_heat_value) / saperatenumber;
	for (int i = 0; i < saperatenumber; i++) 
	{
		double T_old_i = EV->m_old_heat_value + dT_old1 * (0.5 + i);
		double T_new_i = EV->m_heat_value + dT_new1 * (0.5 + i);
		double delta_Ti = T_new_i - T_old_i;
		expansion1 += delta_tr1_len * ThermalExpansionC * delta_Ti;
	}
	double strain1 = expansion1 / truss1_len;
	cv::Point2d Ftruss1 = (-1.0* strain1* YongsModule * Area) * (truss1 / sqrt((truss1).dot(truss1)));

	//truss second
	cv::Point2d truss2 = cv::Point2d(N2->m_pos.x, N2->m_pos.y) - cv::Point2d(EV->m_pos.x, EV->m_pos.y);
	double truss2_len = uniformlengthh(cv::norm(truss2));
	double delta_tr2_len = truss2_len / saperatenumber;
	double dT_old2 = (N2->m_old_heat_value - EV->m_old_heat_value) / saperatenumber;
	double dT_new2 = (N2->m_heat_value - EV->m_heat_value) / saperatenumber;
	double expansion2 = 0; //-1 shrink, 1 compress
	for (int i = 0; i < saperatenumber; i++)
	{
		double T_old_i = EV->m_old_heat_value + dT_old2 * (0.5 + i);
		double T_new_i = EV->m_heat_value + dT_new2 * (0.5 + i);
		double delta_Ti = T_new_i - T_old_i;
		expansion2 += delta_tr2_len * ThermalExpansionC * delta_Ti;
	}
	double strain2 = expansion2 / truss2_len;
	cv::Point2d Ftruss2 = (-1.0 * strain2 * YongsModule * Area) * (truss2 / sqrt((truss2).dot(truss2)));

	//HVD edge info
	HVD::edge* edge = m_vd->m_elist[0][EV->m_edgeindex];
	cv::Point2d ep1, ep2;
	if (edge->IsInfinit())
	{
		if (edge->m_verts[0] == NULL)
		{
			ep1 = cv::Point2d(edge->m_infinit_vert.x, edge->m_infinit_vert.y);
			ep2 = cv::Point2d(edge->m_verts[1]->m_pos.x, edge->m_verts[1]->m_pos.y);
		}
		else
		{
			ep1 = cv::Point2d(edge->m_verts[0]->m_pos.x, edge->m_verts[0]->m_pos.y);
			ep2 = cv::Point2d(edge->m_infinit_vert.x, edge->m_infinit_vert.y);
		}
	}
	else
	{
		ep1 = cv::Point2d(edge->m_verts[0]->m_pos.x, edge->m_verts[0]->m_pos.y);
		ep2 = cv::Point2d(edge->m_verts[1]->m_pos.x, edge->m_verts[1]->m_pos.y);
	}
	edge->m_edgetangent = (ep1 - ep2) / sqrt((ep1 - ep2).dot(ep1 - ep2));
	edge->m_edgenormal = cv::Point2d(edge->m_edgetangent.y, -1.0*edge->m_edgetangent.x);


	//force on the point
	cv::Point2d normal = edge->m_edgenormal * (edge->m_edgenormal.dot(Ftruss1) + edge->m_edgenormal.dot(Ftruss2));
	cv::Point2d tangent = edge->m_edgetangent * (edge->m_edgetangent.dot(Ftruss1) + edge->m_edgetangent.dot(Ftruss2));

	EV->m_force = { normal , tangent };

	m_max_normal_force = std::max(cv::norm(EV->m_force.first), m_max_normal_force);
	m_min_normal_force = std::min(cv::norm(EV->m_force.first), m_min_normal_force);
}

int FDE_Topology::Next_Vert(int vindex, int tindex)  //clock wise //return next vertex in vlist (global)
{
	for (int i = 0; i < 3; i++)
	{
		if (m_vlist[vindex]->m_index == m_tlist[tindex]->m_verts[i]->m_index)
		{
			int next_tri_vert_index = (i + 1) % 3;
			return  m_tlist[tindex]->m_verts[next_tri_vert_index]->m_index;
		}
	}
}

int FDE_Topology::Prev_Vert(int vindex, int tindex)  //counter clock wise //return next vertex in vlist (global)
{
	for (int i = 0; i < 3; i++)
	{
		if (m_vlist[vindex]->m_index == m_tlist[tindex]->m_verts[i]->m_index)
		{
			int next_tri_vert_index = (i + 2) % 3;
			return  m_tlist[tindex]->m_verts[next_tri_vert_index]->m_index;
		}
	}
}

int FDE_Topology::Find_Vert_in_Tri(int vindex, int tindex) //return local index
{
	for (int i = 0; i < 3; i++)
	{
		if (m_vlist[vindex]->m_index == m_tlist[tindex]->m_verts[i]->m_index)
			return i;
	}
}

int FDE_Topology::Find_Edge_in_Tri(int eindex, int tindex) //return local index
{
	for (int i = 0; i < 3; i++)
	{
		if (m_elist[eindex]->m_index == m_tlist[tindex]->m_edges[i]->m_index)
			return i;
	}
}


int* FDE_Topology::Find_3rd_to_6th_cloest_vert(int vindex)
{
	FDE_Triangle* tri = m_vlist[vindex]->m_tris[0];
	int next_vindex_intri = (Find_Vert_in_Tri(vindex, tri->m_index) + 1) % 3;
	int facing_edge_index = tri->m_edges[next_vindex_intri]->m_index;

	//int now_edge_tri0 = m_elist[facing_edge_index]->m_tris[0]->m_index;
	//int now_edge_tri1 = m_elist[facing_edge_index]->m_tris[1]->m_index;
	//int eindex_in_reverse_tri;
	//int now_tri;
	//if (now_edge_tri0 == tri->m_index)
	//{
	//	now_tri = now_edge_tri1;
	//	eindex_in_reverse_tri = Find_Edge_in_Tri(facing_edge_index, now_tri);
	//}
	//else
	//{
	//	now_tri = now_edge_tri0;
	//	eindex_in_reverse_tri = Find_Edge_in_Tri(facing_edge_index, now_tri);
	//}
	//int edge_vert_index = m_tlist[now_tri]->m_verts[eindex_in_reverse_tri]->m_index;
	//int farvert_index = Prev_Vert(edge_vert_index, now_tri);
	//
	//int surroundings[6];
	//surroundings[0] = farvert_index;
	//int index_guard = 1;
	//for (int k = 0; k < 6; k++)
	//{
	//	int index = Next_Vert(m_vlist[farvert_index]->m_index, m_vlist[farvert_index]->m_tris[k]->m_index);
	//	if (index != vindex)
	//	{
	//		surroundings[index_guard] = index;
	//		index_guard += 1;
	//		if (index_guard >= 6)
	//			break;
	//	}
	//}

	int surroundings[6];
	std::vector<int> surroundingvector;
	std::unordered_map<int, int> point_garage; //besides vindex
	int now_edge_vert0 = m_elist[facing_edge_index]->m_verts[0]->m_index;
	int now_edge_vert1 = m_elist[facing_edge_index]->m_verts[1]->m_index;
	point_garage[vindex] += 1;
	point_garage[now_edge_vert0] += 1;
	point_garage[now_edge_vert1] += 1;

	//first distance
	surroundingvector.push_back(now_edge_vert0);
	surroundingvector.push_back(now_edge_vert1);

	//second distance
	int vert0_verts; //iterate end points
	for (auto corner1 : m_vlist[now_edge_vert0]->m_corners)
	{
		if (corner1->m_next->m_oppo == NULL)
		{
			vert0_verts = corner1->m_prev->m_vert->m_index;

			if (point_garage.find(vert0_verts) == point_garage.end())
			{
				surroundingvector.push_back(vert0_verts);
				point_garage[vert0_verts] += 1;
			}
			else
				point_garage[vert0_verts] += 1;

			vert0_verts = corner1->m_next->m_vert->m_index;

			if (point_garage.find(vert0_verts) == point_garage.end())
			{
				surroundingvector.push_back(vert0_verts);
				point_garage[vert0_verts] += 1;
			}
			else
				point_garage[vert0_verts] += 1;
		}
		else
		{
			vert0_verts = corner1->m_next->m_vert->m_index;

			if (point_garage.find(vert0_verts) == point_garage.end())
			{
				surroundingvector.push_back(vert0_verts);
				point_garage[vert0_verts] += 1;
			}
			else
				point_garage[vert0_verts] += 1;
		}
	}

	int vert1_verts; //iterate end points
	for (auto corner2 : m_vlist[now_edge_vert1]->m_corners) 
	{
		if (corner2->m_next->m_oppo == NULL)
		{
			vert1_verts = corner2->m_prev->m_vert->m_index;

			if (point_garage.find(vert1_verts) == point_garage.end())
			{
				surroundingvector.push_back(vert1_verts);
				point_garage[vert1_verts] += 1;
			}
			else
				point_garage[vert1_verts] += 1;

			vert1_verts = corner2->m_next->m_vert->m_index;

			if (point_garage.find(vert1_verts) == point_garage.end())
			{
				surroundingvector.push_back(vert1_verts);
				point_garage[vert1_verts] += 1;
			}
			else
				point_garage[vert1_verts] += 1;
		}
		else
		{
			vert1_verts = corner2->m_next->m_vert->m_index;

			if (point_garage.find(vert1_verts) == point_garage.end())
			{
				surroundingvector.push_back(vert1_verts);
				point_garage[vert1_verts] += 1;
			}
			else
				point_garage[vert1_verts] += 1;
		}
	}

	//thrid distance
	if (surroundingvector.size() < 6)
	{
		int start_point_index = -1;
		for (auto start_point : point_garage)
		{
			if (start_point.second == 1)
			{
				start_point_index = start_point.first;
				break;
			}
		}
		FDE_Corner* corner = m_vlist[start_point_index]->m_corners[0];
		int end_verts; //iterate end points
		for (auto corner : m_vlist[start_point_index]->m_corners)
		{
			end_verts = corner->m_next->m_vert->m_index;

			if (point_garage.find(end_verts) == point_garage.end())
			{
				surroundingvector.push_back(end_verts);
				point_garage[end_verts] += 1;
				if (surroundingvector.size() >= 6)
					break;
			}
			else
				point_garage[end_verts] += 1;
		}
	}
	for (int i = 0; i < 6; i++)
	{
		surroundings[i] = surroundingvector[i];
	}

	return surroundings;
}

int* FDE_Topology::Find_4th_to_6th_cloest_vert(int vindex)
{
	int same_edgevert_index = -1;
	for (auto i : m_vlist[vindex]->m_tris[0]->m_verts)
	{
		for (auto j : m_vlist[vindex]->m_tris[1]->m_verts)
		{
			if (i->m_index == j->m_index
				&& m_vlist[i->m_index]->m_pos.x != 0 && m_vlist[i->m_index]->m_pos.x != m_dim_num - 1 && m_vlist[i->m_index]->m_pos.y != 0 && m_vlist[i->m_index]->m_pos.y != m_dim_num - 1)
			{
				same_edgevert_index = i->m_index;
				break;
			}
		}
	}

	int surroundings[6];
	surroundings[0] = same_edgevert_index;
	int index_guard = 1;
	for (int k = 0; k < 6; k++)
	{
		int index = Next_Vert(m_vlist[same_edgevert_index]->m_index, m_vlist[same_edgevert_index]->m_tris[k]->m_index);
		if (index != vindex)
		{
			surroundings[index_guard] = index;
			index_guard += 1;
			if (index_guard >= 6)
				break;
		}
	}

	return surroundings;
}

int* FDE_Topology::Find_5th_to_6th_cloest_vert_Boundary(int vindex)
{
	int min_index[2] = { -1 };
	int min_dist[2] = { 100000000 };

	for (auto tri : m_vlist[vindex]->m_tris)
	{
		int next_vindex_intri = (Find_Vert_in_Tri(vindex, tri->m_index) + 1) % 3;
		int facing_edge_index = tri->m_edges[next_vindex_intri]->m_index;
		if (m_elist[facing_edge_index]->m_tris.size() == 2)
		{
			int now_edge_tri0 = m_elist[facing_edge_index]->m_tris[0]->m_index;
			int now_edge_tri1 = m_elist[facing_edge_index]->m_tris[1]->m_index;
			int eindex_in_reverse_tri;
			int now_tri;
			if (now_edge_tri0 == tri->m_index)
			{
				now_tri = now_edge_tri1;
				eindex_in_reverse_tri = Find_Edge_in_Tri(facing_edge_index, now_tri);
			}
			else
			{
				now_tri = now_edge_tri0;
				eindex_in_reverse_tri = Find_Edge_in_Tri(facing_edge_index, now_tri);
			}
			int edge_vert_index = m_tlist[now_tri]->m_verts[eindex_in_reverse_tri]->m_index;
			int farvert_index = Prev_Vert(edge_vert_index, now_tri);
			//if (m_vlist[farvert_index]->m_pos.x != 0 && m_vlist[farvert_index]->m_pos.y == 0 && m_vlist[farvert_index]->m_pos.x == m_dim_num - 1 && m_vlist[farvert_index]->m_pos.y == m_dim_num - 1)
			{
				double dist = glm::distance(m_vlist[vindex]->m_pos, m_vlist[farvert_index]->m_pos);
				if (dist < min_dist[0])
				{
					min_index[1] = min_index[0];
					min_index[0] = farvert_index;

				}
				else if (dist >= min_dist[0])
				{
					if (dist < min_dist[1])
						min_index[1] = farvert_index;
				}
			}
		}
	}
	return min_index;
}

int* FDE_Topology::Find_5th_to_6th_cloest_vert(int vindex)
{
	int min_index[2] = {-1};
	int min_dist[2] = { 100000000 };
	for (auto tri : m_vlist[vindex]->m_tris)
	{
		int next_vindex_intri = (Find_Vert_in_Tri(vindex, tri->m_index) + 1) % 3;
		int facing_edge_index = tri->m_edges[next_vindex_intri]->m_index;
		int now_edge_tri0 = m_elist[facing_edge_index]->m_tris[0]->m_index;
		int now_edge_tri1 = m_elist[facing_edge_index]->m_tris[1]->m_index;
		int eindex_in_reverse_tri;
		int now_tri;
		if (now_edge_tri0 == tri->m_index)
		{
			now_tri = now_edge_tri1;
			eindex_in_reverse_tri = Find_Edge_in_Tri(facing_edge_index, now_tri);
		}
		else
		{
			now_tri = now_edge_tri0;
			eindex_in_reverse_tri = Find_Edge_in_Tri(facing_edge_index, now_tri);
		}
		int edge_vert_index = m_tlist[now_tri]->m_verts[eindex_in_reverse_tri]->m_index;
		int farvert_index = Prev_Vert(edge_vert_index, now_tri);
		double dist = glm::distance(m_vlist[vindex]->m_pos, m_vlist[farvert_index]->m_pos);
		if (dist < min_dist[0])
		{
			min_index[1] = min_index[0];
			min_index[0] = farvert_index;
			
		}
		else if (dist >= min_dist[0])
		{
			if (dist < min_dist[1])
				min_index[1] = farvert_index;
		}
	}
	return min_index;
}

int FDE_Topology::Find_6th_cloest_vert_Boundary(int vindex)
{
	int min_index = -1;
	int min_dist = 100000000;
	for (auto corner : m_vlist[vindex]->m_corners)
	{
		if (corner->m_oppo != NULL)
		{
			double dist = glm::distance(m_vlist[vindex]->m_pos, corner->m_oppo->m_vert->m_pos);
			if (dist < min_dist)
			{
				min_index = corner->m_oppo->m_vert->m_index;
				min_dist = dist;
			}
		}
	}
	return min_index;
}

int FDE_Topology::Find_6th_cloest_vert(int vindex)
{
	int min_index = -1;
	int min_dist = 100000000;
	for (auto tri : m_vlist[vindex]->m_tris)
	{
		int next_vindex_intri = (Find_Vert_in_Tri(vindex, tri->m_index) + 1) % 3;
		int facing_edge_index = tri->m_edges[next_vindex_intri]->m_index;
		int now_edge_tri0 = m_elist[facing_edge_index]->m_tris[0]->m_index;
		int now_edge_tri1 = m_elist[facing_edge_index]->m_tris[1]->m_index;
		int eindex_in_reverse_tri;
		int now_tri;
		if (now_edge_tri0 == tri->m_index)
		{
			now_tri = now_edge_tri1;
			eindex_in_reverse_tri = Find_Edge_in_Tri(facing_edge_index, now_tri);
		}
		else
		{
			now_tri = now_edge_tri0;
			eindex_in_reverse_tri = Find_Edge_in_Tri(facing_edge_index, now_tri);
		}
		int edge_vert_index = m_tlist[now_tri]->m_verts[eindex_in_reverse_tri]->m_index;
		int farvert_index = Prev_Vert(edge_vert_index, now_tri);
		double dist = glm::distance(m_vlist[vindex]->m_pos, m_vlist[farvert_index]->m_pos);
		if (dist < min_dist)
		{
			min_index = farvert_index;
			min_dist = dist;
		}
	}
	return min_index;

	//std::unordered_map<int, int> point_garage; //besides vindex
	//point_garage[vindex] += 1;
	//for (auto corner : m_vlist[vindex]->m_corners)
	//{
	//	point_garage[corner->m_next->m_vert->m_index] += 1;
	//}

}

//================================================================================================================

void FDE_Topology::DrawGradLine(cv::Mat canvas, cv::Point pt1, cv::Point pt2, cv::Vec3f color1, cv::Vec3f color2)
{
	// see https://docs.opencv.org/master/dc/dd2/classcv_1_1LineIterator.html
	cv::LineIterator it( canvas, pt1, pt2, cv::LINE_8);

	for (int i = 0; i < it.count; ++i, ++it)
	{
		float alpha = i / (float)(it.count - 1); // 0..1 along the line

		cv::Vec3b blended = color1 * (1 - alpha) + color2 * (alpha);

		//if (it.pos().x != 0 && it.pos().y != 0 && it.pos().x != canvas.cols - 1 && it.pos().y != canvas.rows - 1)
		//{
		//	canvas.at<cv::Vec3b>(cv::Point(it.pos().x - 1, it.pos().y + 1)) = blended;
		//	canvas.at<cv::Vec3b>(cv::Point(it.pos().x + 1, it.pos().y - 1)) = blended;
		//	canvas.at<cv::Vec3b>(cv::Point(it.pos().x - 1, it.pos().y - 1)) = blended;
		//	canvas.at<cv::Vec3b>(cv::Point(it.pos().x + 1, it.pos().y + 1)) = blended;
		//	canvas.at<cv::Vec3b>(cv::Point(it.pos().x - 1, it.pos().y)) = blended;
		//	canvas.at<cv::Vec3b>(cv::Point(it.pos().x + 1, it.pos().y)) = blended;
		//	canvas.at<cv::Vec3b>(cv::Point(it.pos().x, it.pos().y - 1)) = blended;
		//	canvas.at<cv::Vec3b>(cv::Point(it.pos().x, it.pos().y + 1)) = blended;
		//}
		canvas.at<cv::Vec3b>(it.pos()) = blended;
		
	}
}

std::vector<std::pair<std::vector<cv::Point2i>, std::vector<cv::Point3f>>> FDE_Topology::DrawTempMap()
{
	std::vector<std::pair<std::vector<cv::Point2i>, std::vector<cv::Point3f>>> output;

	std::vector<cv::Point2i> point_garage0;
	std::vector<cv::Point3f> point_color_garage0;
	for (auto e : m_elist)
	{
		cv::Point2i p1 = cv::Point2i(e->m_verts[0]->m_pos.x, e->m_verts[0]->m_pos.y);
		cv::Point2i p2 = cv::Point2i(e->m_verts[1]->m_pos.x, e->m_verts[1]->m_pos.y);
		
		double c1 = (e->m_verts[0]->m_heat_value - 273.15) / abs(313.15 - 273.15);
		double c2 = (e->m_verts[1]->m_heat_value - 273.15) / abs(313.15 - 273.15);
		
		QVector3D d1j = color_t_Jet(c1);
		QVector3D d2j = color_t_Jet(c2);
		point_garage0.push_back(p1);
		point_garage0.push_back(p2);
		point_color_garage0.push_back(cv::Point3f(d1j.x(), d1j.y(), d1j.z()));
		point_color_garage0.push_back(cv::Point3f(d2j.x(), d2j.y(), d2j.z()));
	}

	std::vector<cv::Point2i> point_garage1;
	std::vector<cv::Point3f> point_color_garage1;

	for (auto t : m_tlist)
	{
		for (auto v : t->m_verts)
		{
			cv::Point2i pv = cv::Point2i(v->m_pos.x,v->m_pos.y);
			double c = (v->m_heat_value - 273.15) / abs(313.15 - 273.15);
			QVector3D d1j = color_t_Jet(c);
			point_garage1.push_back(pv);
			point_color_garage1.push_back(cv::Point3f(d1j.x(), d1j.y(), d1j.z()));
		}
	}

	std::vector<cv::Point2i> point_garage2;
	std::vector<cv::Point3f> point_color_garage2;

	for (int i = 0; i < m_vd->m_clist[0].size(); i++)
	{
	    std::vector<cv::Point2i> points;
	    std::vector<cv::Point3f> point_colors;
	
	    float eta = 40.0 * (m_vd->m_cell_alpha_value[0][i] + 0.00008);
	    if (eta > 1.0)  eta = 1.0;
	    QVector3D color = color_t_Jet(eta);
		cv::Point2i p = cv::Point2i(m_vd->m_clist[0][i]->m_site_pos.x, m_vd->m_clist[0][i]->m_site_pos.y);
	    cv::Point3f C = cv::Point3f(color.x(), color.y(), color.z());
	
	    for (auto edge : m_vd->m_clist[0][i]->m_edges)
	    {
			cv::Point2i p0, p1;
	
	        if (edge->m_verts[0] == NULL)
	        {
	            p0 = cv::Point2i(edge->m_infinit_vert.x, edge->m_infinit_vert.y);
	            p1 = cv::Point2i(edge->m_verts[1]->m_pos.x, edge->m_verts[1]->m_pos.y);
	        }
	        else if (edge->m_verts[1] == NULL)
	        {
	            p0 = cv::Point2i(edge->m_verts[0]->m_pos.x, edge->m_verts[0]->m_pos.y);
	            p1 = cv::Point2i(edge->m_infinit_vert.x, edge->m_infinit_vert.y);
	        }
	        else
	        {
	            p0 = cv::Point2i(edge->m_verts[0]->m_pos.x, edge->m_verts[0]->m_pos.y);
	            p1 = cv::Point2i(edge->m_verts[1]->m_pos.x, edge->m_verts[1]->m_pos.y);
	        }
			point_garage2.push_back(p);
			point_garage2.push_back(p0);
			point_garage2.push_back(p1);
			point_color_garage2.push_back(C);
			point_color_garage2.push_back(C);
			point_color_garage2.push_back(C);
	    }
	}

	output.push_back({ point_garage0, point_color_garage0 });
	output.push_back({ point_garage1, point_color_garage1 });
	output.push_back({ point_garage2, point_color_garage2 });

	return output;
}

void FDE_Topology::OutPutStressExcel()
{
	auto it = m_DicHVDEtoTriv.begin();
	for (int i = 0; i < 5; i++)
	{
		std::fstream outf;
		std::string s0("./TempMap/Heat_Experiment_4/");
		s0.append(std::to_string(it->second));
		s0.append("point.csv");
		outf.open(s0, std::ios::app);
		if (!outf.is_open())
			qDebug() << "Error when do out put";

		outf.seekg(0, std::ios::end); // put the "cursor" at the end of the file
		int length = outf.tellg();

		//if (length == 0)
		//{
		//	outf.clear();
		//	outf << "P1 norm-shear" << "," << "P2 norm-shear" << "," << "P3 norm-shear" << "," << "P4 norm - shear" << "," << "P5 norm - shear" << std::endl;
		//}
		//else
		//{

		//temperature, delta_temp, norm x, norm y, shear x, shear y
		int vindex = it->second;
		outf << m_vlist[vindex]->m_heat_value << "," << m_vlist[vindex]->m_heat_value - m_vlist[vindex]->m_old_heat_value << "," 
			<< m_vlist[vindex]->m_force.first.x << "," << m_vlist[vindex]->m_force.first.y << ","
			<< m_vlist[vindex]->m_force.second.x << "," << m_vlist[vindex]->m_force.second.y << std::endl;
		std::advance(it, 50);
		outf.close();
	}
}
