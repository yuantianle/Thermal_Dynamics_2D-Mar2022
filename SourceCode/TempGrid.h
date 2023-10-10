#pragma once

#include <vector>
#include <qdebug.h>

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

#include <unordered_map>

#include <boost/polygon/voronoi.hpp>
using boost::polygon::voronoi_builder;
using boost::polygon::voronoi_diagram;


typedef Eigen::SparseMatrix<double> SpMat;  //declare a double sparse matrix type
typedef Eigen::Triplet<double> T; //triple:(row, column, value)

typedef glm::vec2 vec2;
typedef glm::vec3 vec3;


class TempGridPoint  // this class is for the new temperature map
{

public:
	TempGridPoint() :pos(0), vector(0), heat_value(0), constrainflag(false) {}

	TempGridPoint(vec2 position): vector(0), heat_value(0), constrainflag(false) 
	{ 
		pos = position; 
	}
	TempGridPoint(vec2 position, double heat): vector(0), constrainflag(false)
	{ 
		pos = position; 
		heat_value = heat; 
	}
	~TempGridPoint() {}

private:
	vec2 pos;
	vec3 vector;
	double heat_value;
	double heat_old;
	bool constrainflag;
	double point_alpha; // save the thermal coefficient of each point
	bool damageholeflag = false;

public:
	vec2 ReturnPos() { return pos; }

public:
	void SetTemp(double heat) { heat_value = heat; }
	void SetOldTemp(double heat) { heat_old = heat; }
	double ReturnTemp() { return heat_value; }
	double ReturnOldTemp() { return heat_old; }
	void SetConsFlag() { constrainflag = true; }
	bool ReturnConsFlag() { return constrainflag; }
	void SetAlpha(double a) { point_alpha = a; }
	double ReturnAlpha() { return point_alpha; }
	void SetDamageHoleflag(bool b) { damageholeflag = b; }
	bool ReturnDamageHoleflag() { return damageholeflag; }
};

class TempVectorPoint  // this class is for the new temperature vector map
{
public:
	TempVectorPoint(vec2 p)
	{
		pos = p;
		TransformMatrix = QMatrix4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
	}
	~TempVectorPoint() {}

private:
	vec2 pos;
	QMatrix4x4 TransformMatrix;

public:
	vec2 ReturnPos() { return pos; }
	QMatrix4x4 ReturnTMatix() { return TransformMatrix; }
	void SetTMatrix(QMatrix4x4 T) { TransformMatrix = T; }
};


class TempGrid
{
public:
	TempGrid(double col, double row, int num);  // col length, row length, number of points
	~TempGrid();

private:
	int m_texturesize_row = 1000;
	int m_texturesize_col = 1000;
	cv::Mat m_TempMap;
	cv::Mat m_ContourMap;
	cv::Mat m_VectorMap;
	QOpenGLTexture* m_Tex_Temperature;
	QOpenGLTexture* m_Tex_Temperature_heat;

	double m_width, m_height;
	int m_dim_num;
	double m_unit_lenx; //X direction unit len
	double m_unit_leny;
	int m_numPoints;
	std::vector<TempGridPoint> m_Points; // save all grid points in the grid garage
	int m_Boundary_Status; //
	//0. puretop
	//1. pureleft
	//2. pureright
	//3. topleft
	//4. topright

	double m_Temp_max = 20, m_Temp_min = 0;


	double m_Field_dense = 40;
	std::vector<TempVectorPoint> m_VectorPoints;
	//std::vector<std::pair<double, double>> m_GradientGarage;
	bool m_contour_flag = false;


public:
	double ReturnWidth() { return m_width; }
	double ReturnHeight() { return m_height; }
	double ReturnMaxTemp() { return m_Temp_max; }
	double ReturnMinTemp() { return m_Temp_min; }
	int ReturnDimNum() { return m_dim_num; }
	double ReturnUnitLenX() { return m_unit_lenx; }
	double ReturnUnitLenY() { return m_unit_leny; }
	void InitializeGrid();
	void SetHeatBoundary(int row, int col, double temp);
	void SetHeat(int row, int col, double temp);
	void SetHeatDamage(int row, int col, double temp);
	void SetHeatCoefficient(std::vector<vec3> alpha_list, voronoi_diagram<double>* vor, vec2 origin_dim);
	void UpdateHeat();
	void UpdateHeat_implicit(double step_length);
	void SetBoundaryStatus(int i) { m_Boundary_Status = i; }
	std::pair<double, double> CalculateGradient(double x, double y);

	QOpenGLTexture* ReturnTextureTemp() { return m_Tex_Temperature;	}
	QOpenGLTexture* ReturnTextureTempH() { return m_Tex_Temperature_heat; }


	std::vector<TempGridPoint> ReturnGrid();
	std::vector<TempVectorPoint> ReturnVectorGrid();
	void UpdateVFieldDense(int Dense);
	double GetTemp(double x, double y);

	void UpdateTemerature();
	void UpdateVecterField();
	void UpdateContourField();

	TempGridPoint& operator[](int idx) { return m_Points[idx]; } // To get the grid point according to it's index
	TempGridPoint& operator()(int row, int col) { return m_Points[col + row * m_dim_num]; } // To get the grid point according to its 3D index
};