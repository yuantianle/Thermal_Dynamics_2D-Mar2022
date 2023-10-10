#pragma once
#include <vector>
#include <qdebug.h>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#include <QImage>
#include <QOpenGLTexture>
#include <QMatrix4x4>

#include "Rainbow.h"

#include <unordered_map>

#include <boost/polygon/voronoi.hpp>

#include <HVD.h>

using boost::polygon::voronoi_builder;
using boost::polygon::voronoi_diagram;


typedef Eigen::SparseMatrix<double> SpMat;  //declare a double sparse matrix type
typedef Eigen::Triplet<double> T; //triple:(row, column, value)

typedef glm::vec2 vec2;
typedef glm::vec3 vec3;

class FDE_Triangle;
class FDE_Edge;
class FDE_Corner;

class FDE_Vertex
{
public:
	FDE_Vertex(vec2 p) : m_index(-1), m_neighborweight(0) { m_pos.x = p.x; m_pos.y = p.y; }
	FDE_Vertex(double xx, double yy) : m_index(-1), m_neighborweight(0) { m_pos.x = xx; m_pos.y = yy; }
	~FDE_Vertex() { }

public:
	bool m_isonVert = false; // if it is true it is point on vertex

	int m_edgeindex = -1; //if it is non -1 it is point on HVD edge; index is HVD edge index;
	std::pair<cv::Point2d, cv::Point2d> m_force; //first normal, second share

	//heat
	double m_heat_value = 0;
	double m_old_heat_value = 273.15;

	//topology
	int m_index; //index of the vertex in FDE topology
	int m_site_index = -1; // [0 ~m_site.size()] is the same index of Voronoi Diagram sites
	vec2 m_pos;
	std::vector<FDE_Triangle*> m_tris;
	std::vector<FDE_Corner*> m_corners;
	std::vector<double> m_neighborweight;

	//double m_heat_old = -1;
	bool m_constrainflag = false;
	double m_point_alpha = -1; // save the thermal coefficient of each point

public:
	std::pair<int, int> m_cell_sites_dic = { -1, -1 }; //index is sites index, for the vertex who locate on edge
	// when  m_cell_sites_dic != { -1, -1 }, the vertex have information shown below
	
	double m_thermal_strain = 0;
	double m_thermal_expansion_coef = 0.00055; //the coefficient of concret 

public:
	int ntris() { return (int)m_tris.size(); }
	int ncorners() { return (int)m_corners.size(); }
};

class FDE_Edge
{
public:
	FDE_Edge() : m_index(-1), m_length(-1) { m_verts[0] = NULL; m_verts[1] = NULL; }
	~FDE_Edge() {}

public:
	double m_length;
	int m_index; 
	FDE_Vertex* m_verts[2];
	std::vector<FDE_Corner*> m_corners;
	int m_ntris;
	std::vector<FDE_Triangle*> m_tris; 

public:
	int ntris() { return (int)m_tris.size(); }
};

class FDE_Corner
{
public:

	FDE_Triangle* m_tri;
	FDE_Vertex* m_vert;
	FDE_Edge* m_edge;

	int m_eDir;
	FDE_Corner* m_prev;
	FDE_Corner* m_next;
	FDE_Corner* m_oppo;

	double m_mean_value_coordinate;//weight
public:
	FDE_Corner() : m_prev(NULL), m_next(NULL), m_oppo(NULL), m_tri(NULL), m_vert(NULL), m_edge(NULL), m_mean_value_coordinate(0.0) {}
};

class FDE_Triangle
{
public:
	FDE_Triangle() : m_index(-1), m_area(-1) 
	{
		m_verts[0] = NULL; m_verts[1] = NULL; m_verts[2] = NULL;
		m_edges[0] = NULL; m_edges[1] = NULL; m_edges[2] = NULL;
		m_corners[0] = NULL; m_corners[1] = NULL; m_corners[2] = NULL;
	}
	FDE_Triangle(FDE_Vertex* v0, FDE_Vertex* v1, FDE_Vertex* v2) : m_index(-1), m_area(-1)
	{
		m_verts[0] = v0; m_verts[1] = v1; m_verts[2] = v2;
		m_edges[0] = NULL; m_edges[1] = NULL; m_edges[2] = NULL;
		m_corners[0] = NULL; m_corners[1] = NULL; m_corners[2] = NULL;
	}
	~FDE_Triangle() {}
public:
	double m_area;
	int m_index;
	FDE_Vertex* m_verts[3];
	FDE_Edge* m_edges[3];
	FDE_Corner* m_corners[3];
};

class FDE_Topology
{
public:
	FDE_Topology(HVD* vd, std::vector<cv::Point>* sites, int row, int column, std::unordered_map<int, int> boundary_flag);
	~FDE_Topology(){ finalize(); }

public:

	HVD* m_vd;
	std::vector<cv::Point>* m_sites;
	//std::unordered_map<int, double> m_alpha_list; // the index will be the same as the m_vlist (vd sites' index)
	std::unordered_map<int, int> m_boundary_flag;
	int m_boundary_status = 4; //0. puretop //1. pureleft //2. pureright //3. topleft //4. topright
	std::vector<FDE_Triangle*> m_tlist;/* list of triangles */
	std::vector<FDE_Vertex*>   m_vlist;/* list of vertices */
	std::vector<FDE_Edge*>     m_elist;/* list of edges */
	std::vector<FDE_Corner*>   m_clist;/* list of corners */
	cv::Mat m_Vtexture;
	cv::Mat m_VDtexture;
	cv::Mat m_Dtexture;
	cv::Mat m_TempMapJet;
	cv::Mat m_TempMapHeat;
	QOpenGLTexture* m_Tex_Temperature;
	QOpenGLTexture* m_Tex_Temperature_heat;
	int m_canvas_row;
	int m_canvas_column;
	int m_dim_num = 1000; //square pixel map length (for thermal visualization)

public://for stress testing
	double m_max_normal_force = -1;
	double m_min_normal_force = 100000;

public:
	std::vector<FDE_Vertex*> m_BreakPointGarage;
	std::unordered_map<int, int> m_DicBreakPoint;

	std::unordered_map<int, int> m_DicHVDCtoTriv;  //point on HVD_m_clist to triangle mesh vertex_index;
	std::unordered_map<int, int> m_DicHVDEtoTriv;  //point on HVD_m_elist to triangle mesh vertex_index;
	std::unordered_map<int, int> m_DicHVDVtoTriv;  //point on HVD_m_vlist to triangle mesh vertex_index;


public:
	int ntris() { return (int)m_tlist.size(); }
	int nverts() { return (int)m_vlist.size(); }
	int nedges() { return (int)m_elist.size(); }
	int ncorners() { return (int)m_clist.size(); }

	// initialization and finalization
	void initialize();
	void finalize();

private:
	void construction();
	void update_properties() {}
	void constructEdges();
	void constructCorners();

	void create_edge(FDE_Vertex*, FDE_Vertex*);
	void order_vertex_to_tri_ptrs(FDE_Vertex*);
	void vertex_to_tri_ptrs();
	FDE_Triangle* find_oppsite_triangle(FDE_Triangle*, FDE_Vertex*, FDE_Vertex*);
	FDE_Triangle* find_oppsite_triangle(FDE_Triangle*, FDE_Edge*);

	int* Find_3rd_to_6th_cloest_vert(int vindex);
	int* Find_4th_to_6th_cloest_vert(int vindex);
	int* Find_5th_to_6th_cloest_vert_Boundary(int vindex);
	int* Find_5th_to_6th_cloest_vert(int vindex);
	int Find_6th_cloest_vert_Boundary(int vindex); //vertex incident 5 tris
	int Find_6th_cloest_vert(int vindex); //vertex incident 4 tris
	int Next_Vert(int vindex, int tindex);
	int Prev_Vert(int vindex, int tindex);
	int Find_Vert_in_Tri(int vindex, int tindex);
	int Find_Edge_in_Tri(int eindex, int tindex);

public:
	void RenderVDMesh();
	void RenderDMesh();
	int ReturnWidth() { return m_canvas_row; }
	int ReturnHeight() { return m_canvas_column;}
	int ReturnDimNum() { return m_dim_num; }

public:
	//----------Function for heat transfer----------------
	void SetHeatCoefficient(std::vector<double>* sites_alpha);
	void InitializeGrid();
	void SetHeat(int index, double temp); // temp initialization
	void SetHeatBoundary(int index, double temp); // temp initialization
	void SetBoundaryStatus(int i) { m_boundary_status = i; }

	double m_max_temp = 313.15;
	double m_min_temp = 273.15;
	
	void UpdateHeat_implicit(double step_length);
	void UpdateEdgeStress();
	void CalculateStress(std::vector<int> nuclvert_vindex, int edgepoint_vindex);
	
	double* CalculateCoefficient(double dt, double alpha, int coreindex, int* surroundings);
	double* CalculateCoefficientBoundaryX(double dt, double alpha, int coreindex, int* surroundings);
	double* CalculateCoefficientBoundaryY(double dt, double alpha, int coreindex, int* surroundings);
	std::vector<std::pair<std::vector<cv::Point2i>, std::vector<cv::Point3f>>> DrawTempMap();
	void DrawGradLine(cv::Mat canvas, cv::Point pt1, cv::Point pt2, cv::Vec3f color1, cv::Vec3f color2);
	QOpenGLTexture* ReturnTextureTemp() { return m_Tex_Temperature; }
	QOpenGLTexture* ReturnTextureTempH() { return m_Tex_Temperature_heat; }
	std::vector<FDE_Vertex*>* ReturnBreakPoints() { return &m_BreakPointGarage; }
	
	//-----------------------------------------------------
	void OutPutStressExcel();
};