#include "TempGrid.h"
#include <opencv2/core/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <QVector>
#include <QVector3D>
#include "Random.h"
#include "Rainbow.h"
#include <boost/polygon/voronoi.hpp>
#include <sstream>
#include <iomanip>

using boost::polygon::voronoi_builder;
using boost::polygon::voronoi_diagram;

typedef glm::vec2 vec2;
typedef glm::vec3 vec3;

TempGrid::TempGrid(double col , double row, int num)
{
	m_width = col; //X direction
	m_height = row; //Y direction
	m_numPoints = num; //num has to be the square number
}

TempGrid::~TempGrid() 
{
	if (m_Tex_Temperature)
		delete m_Tex_Temperature;
}


void TempGrid::InitializeGrid()
{
	m_dim_num = sqrt(m_numPoints);
	m_unit_lenx = m_width / (m_dim_num-1); //X direction unit len
	m_unit_leny = m_height / (m_dim_num-1);
	double startx = -0.5 * m_width;
	double starty = 0.5 * m_height;

	for (int j = 0; j < m_dim_num; j++) // row
	{
		for (int i = 0; i < m_dim_num; i++)  // colum
		{
			vec2 pos = vec2(startx + i * m_unit_lenx, starty - j * m_unit_leny);
			TempGridPoint tpoint;
			if (i == m_dim_num - 1 ||  j == 0) tpoint = TempGridPoint(pos, 313.15);
			else if (i == 0 || j == m_dim_num - 1) tpoint = TempGridPoint(pos, 273.15);
			else tpoint = TempGridPoint(pos, 0);
			m_Points.push_back(tpoint);
		}
	}

	double unit_x = 1 / (m_Field_dense);
	double unit_y = 1 / (m_Field_dense);
	for (int row = 0; row < m_Field_dense; row++) //row
	{
		for (int col = 0; col < m_Field_dense; col++) //col
		{
			TempVectorPoint vp(vec2(-0.5 + (0.5* unit_x) + col* unit_x, 0.5 - (0.5 * unit_y) - row* unit_y));
			m_VectorPoints.push_back(vp);
		}
	}
}

void TempGrid::UpdateVFieldDense(int Dense)
{
	m_VectorPoints.erase(m_VectorPoints.begin(), m_VectorPoints.end());
	m_Field_dense = Dense;
	double unit_x = 1 / (m_Field_dense);
	double unit_y = 1 / (m_Field_dense);
	for (int row = 0; row < m_Field_dense; row++) //row
	{
		for (int col = 0; col < m_Field_dense; col++) //col
		{
			TempVectorPoint vp(vec2(-0.5 + (0.5 * unit_x) + col * unit_x, 0.5 - (0.5 * unit_y) - row * unit_y));
			m_VectorPoints.push_back(vp);
		}
	}
	UpdateVecterField();
}

void TempGrid::SetHeatBoundary(int row, int col, double temp)
{
	m_Points[(row)*m_dim_num + col].SetTemp(temp);
	m_Temp_max = std::max(temp, m_Temp_max);
	m_Temp_min = std::min(temp, m_Temp_min);
}

void TempGrid::SetHeat(int row, int col, double temp)
{
	m_Points[(row) *m_dim_num + col].SetTemp(temp);
	m_Points[(row) *m_dim_num + col].SetConsFlag();
	m_Temp_max = std::max(temp, m_Temp_max);
	m_Temp_min = std::min(temp, m_Temp_min);
}

void TempGrid::SetHeatDamage(int row, int col, double temp)
{
	m_Points[(row)*m_dim_num + col].SetTemp(temp);
	m_Points[(row)*m_dim_num + col].SetConsFlag();
	m_Temp_max = std::max(temp, m_Temp_max);
	m_Temp_min = std::min(temp, m_Temp_min);
	m_Points[(row)*m_dim_num + col].SetDamageHoleflag(true);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////
const double eps = 1e-6;
const double PI = acos(-1);

//compare two double's size under eps pricise
int dcmp(double x)
{
	if (fabs(x) < eps) return 0;
	else
		return x < 0 ? -1 : 1;
}
//Judge whether point Q is on the segment between P1 and P2
bool OnSegment(vec2 P1, vec2 P2, vec2 Q)
{
	//The first judge: Q on P1P2 line
	//The second judge: In the range between P1 and P2
	double d;
	vec2 n1 = (P1 - Q);
	vec2 n2 = (P2 - Q);
	d = n1.x * n2.y - n2.x * n1.y;
	return dcmp(d) == 0 && dcmp(dot((P1 - Q),(P2 - Q))) <= 0;
}
//The algorithm judge the point P in the polygon -- The ray method
bool InPolygon(vec2 P, std::vector<vec2> vertices)
{
	bool flag = false; //calculate odd or even
	vec2 P1, P2; //two vertices of an edge
	int n = vertices.size();
	for (int i = 0, j = n-1; i < n; j = i++)
	{
		P1 = vertices[i];
		P2 = vertices[j];
		if (OnSegment(P1, P2, P)) return true; 
		if ((dcmp(P1.y - P.y) > 0 != dcmp(P2.y - P.y) > 0) && dcmp(P.x - (P.y - P1.y) * (P1.x - P2.x) / (P1.y - P2.y) - P1.x) < 0)
			flag = !flag;
	}
	return flag;
}
/////////////////////////////////////////

void DrawRulerColor(cv::Mat m)
{
	int xmove = 50;
	cv::circle(m, cv::Point2i(19 + xmove, 439), 3, cv::Scalar(0, 0, 0), -1);
	
	putText(m, std::to_string(0), cv::Point2i( xmove - 10, 439), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2, false);
	
	cv::Mat colorbar = cv::imread("./CrackMaps/bar3.png");
	cv::resize(colorbar, colorbar, cv::Size(25, 420));
	cv::Mat dst_roi = m(cv::Rect(xmove + 20, 19, colorbar.cols, colorbar.rows));
	colorbar.copyTo(dst_roi);
	
	for (int y = 19; y < 429; y += 5)
	{
		if ((y + 1) % 40 == 10)
		{
			cv::line(m, cv::Point2i(19+xmove, y), cv::Point2i(30+ xmove, y), cv::Scalar(0, 0, 0), 1);
			double value = 25 - 2.272 * (float)((y + 1) - 10) / 40;
			std::stringstream stream;
			stream << std::fixed << std::setprecision(2) << value;
			std::string s = stream.str();
			putText(m, s, cv::Point2i(xmove - 30, y + 4), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2, false);
		}
	}
	cv::arrowedLine(m, cv::Point2i(19+ xmove, 439), cv::Point2i(19+ xmove, 19), cv::Scalar(0, 0, 0), 2, 8, 0, 0.015);
	putText(m, "Alpha value", cv::Point2i(xmove -15, 465), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2, false);
	putText(m, "10e-6 (m^2)/s", cv::Point2i(xmove - 20, 490), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2, false);
}

void TempGrid::SetHeatCoefficient(std::vector<vec3> alpha_list, voronoi_diagram<double>* vor, vec2 origin_dim)
{
	//this step is for check whether the temp grid point locate 
	std::vector<std::pair<std::vector<vec2>,double>> elementPoints_garage;
	for (voronoi_diagram<double>::const_cell_iterator it = vor->cells().begin();
		it != vor->cells().end(); ++it)
	{
		const voronoi_diagram<double>::cell_type& cell = *it;
		vec3 a = alpha_list[cell.source_index()];

		//--------------------------------------To get all of the verticies---------------------
		const voronoi_diagram<double>::edge_type* edge = cell.incident_edge();
		std::vector<vec2> elementPoints;
		if (edge->is_finite() && edge->vertex0()->x() >= 0 && edge->vertex0()->y() >= 0)
		{
			elementPoints.push_back(vec2(edge->vertex0()->x() / origin_dim.y * m_width - 0.5 * m_width, (origin_dim.x - edge->vertex0()->y()) / origin_dim.x * m_height - 0.5 * m_height)); // record the vertex
		}
		do {
			edge = edge->next();
			if (edge->is_finite())//&& edge->vertex0()->x() >= 0 && edge->vertex0()->y() >= 0
			{
				double y = origin_dim.x - edge->vertex0()->y();
				y = y / origin_dim.x;
				y = y * m_height;
				y = y - 0.5 * m_height;
				double x = edge->vertex0()->x();
				x = x / origin_dim.x;
				x = x * m_width;
				x = x - 0.5 * m_width;

				elementPoints.push_back(vec2(edge->vertex0()->x() / origin_dim.y * m_width - 0.5 * m_width, (origin_dim.x - edge->vertex0()->y()) / origin_dim.x * m_height - 0.5 * m_height)); // record the vertex
			}
		} while (edge != cell.incident_edge());
		//--------------------------------------To get all of the verticies---------------------

		elementPoints_garage.push_back(std::make_pair(elementPoints, a.x));
	}

	for (int i = 0; i < m_Points.size(); i++)
	{
		m_Points[i].SetAlpha(0.01);
		double alpha;

		for (auto k: elementPoints_garage)
		{
			vec2 point = m_Points[i].ReturnPos();

			if (InPolygon(point, k.first))
			{
				alpha = k.second;
				break;
			}
		}

		//double alpha = (Random::Float()) * 0.05;
		m_Points[i].SetAlpha(alpha);
	}

	cv::Mat src = cv::Mat::zeros(m_dim_num, m_dim_num, CV_8UC3); 
	src.setTo(255);

	uchar* pxvec = src.ptr<uchar>(0);
	for (int i = 0; i < src.rows; i++)
	{
		pxvec = src.ptr<uchar>(i);

		for (int j = 0; j < src.cols; j++)
		{
			double alpha = m_Points[i * m_dim_num + j].ReturnAlpha();
			float eta = 40.0 * (alpha + 0.00008);
			if (eta > 1.0)  eta = 1.0;
			QVector3D d1 = 255 * color_t_Jet(eta);
			pxvec[3 * j + 0] = d1.z();
			pxvec[3 * j + 1] = d1.y();
			pxvec[3 * j + 2] = d1.x();
		}
	}

	cv::resize(src, src, cv::Size(m_texturesize_row, m_texturesize_col), cv::INTER_CUBIC);
	DrawRulerColor(src);

	cv::imwrite("./CrackMaps/Texture_Alpha.png", src);

}
/////////////////////////////////////////////////////////////////////////////////////////////////////////

void TempGrid::UpdateHeat_implicit(double step_length)
{
	double heat_flux_C = 1;

	double dt = step_length;//50
	int m = m_Points.size();
	//double alpha = 0.005;
	double dx = m_unit_lenx;
	double dy = m_unit_leny;
	//double rx = alpha * dt / pow(dx, 2);
	//double ry = alpha * dt / pow(dy, 2);

	//1. Assembly:
	std::vector<T> tripletList;   // list of non-zeros coefficients the triple value
	Eigen::VectorXd b(m);     // b vector at the right side of the equation

	//2. BuildProblem: push back coefficient
	b.setZero();

	for (int i = 0; i < m_Points.size(); i++)
	{
		int row = i / m_dim_num;
		int col = i % m_dim_num;

		double alpha = m_Points[i].ReturnAlpha();
		double rx = alpha * dt / pow(dx, 2);
		double ry = alpha * dt / pow(dy, 2);

		b[i] = m_Points[i].ReturnTemp();

		if (row == 0 || col == 0 || row == m_dim_num - 1 || col == m_dim_num - 1)
		{
			
			//if ((row == 0 || col == m_dim_num - 1) && row != m_dim_num - 1 && col != 0)
			//	tripletList.push_back(T(i, i, 1));
			//else if (col == 0 && row != m_dim_num - 1)
			//{
			//	tripletList.push_back(T(i, i, 1 + 2 * rx));
			//	tripletList.push_back(T(i, row * m_dim_num + (col + 1), - 2 * rx));
			//	b[i] -= 2 * rx * dx * heat_flux_C;
			//}
			//else
			//{
			//	tripletList.push_back(T(i, i, 1 + 2 * ry));
			//	tripletList.push_back(T(i, (row - 1) * m_dim_num + col, -2 * ry));
			//	b[i] -= 2 * ry * dy * heat_flux_C;
			//}
			// 
			//0. puretop
			if (m_Boundary_Status == 0)
			{
				if (row == 0 && col != 0 && col != m_dim_num - 1)
				{
					tripletList.push_back(T(i, i, 1));
				}
				else if (col == 0 && row != m_dim_num - 1)
				{
					tripletList.push_back(T(i, i, 1 + 2 * rx));
					tripletList.push_back(T(i, row * m_dim_num + (col + 1), - 2 * rx));
					b[i] -= 2 * rx * dx * heat_flux_C;
				}
				else if (col == m_dim_num - 1 && row != m_dim_num - 1)
				{
					tripletList.push_back(T(i, i, 1 + 2 * rx));
					tripletList.push_back(T(i, row * m_dim_num + (col - 1), - 2 * rx));
					b[i] -= 2 * rx * dx * heat_flux_C;
				}
				else
				{
					tripletList.push_back(T(i, i, 1 + 2 * ry));
					tripletList.push_back(T(i, (row - 1) * m_dim_num + col, -2 * ry));
					b[i] -= 2 * ry * dy * heat_flux_C;
				}
			}
			//1. pureleft
			else if (m_Boundary_Status == 1)
			{
				if (col == 0 && row != 0 && row != m_dim_num - 1)
				{
					tripletList.push_back(T(i, i, 1));
				}
				else if (row == 0 && col != m_dim_num - 1)
				{
					tripletList.push_back(T(i, i, 1 + 2 * ry));
					tripletList.push_back(T(i, (row + 1) * m_dim_num + col, -2 * ry));
					b[i] -= 2 * ry * dy * heat_flux_C;
				}
				else if (col == m_dim_num - 1 && row != m_dim_num - 1)
				{
					tripletList.push_back(T(i, i, 1 + 2 * rx));
					tripletList.push_back(T(i, row * m_dim_num + (col - 1), -2 * rx));
					b[i] -= 2 * rx * dx * heat_flux_C;
				}
				else
				{
					tripletList.push_back(T(i, i, 1 + 2 * ry));
					tripletList.push_back(T(i, (row - 1) * m_dim_num + col, -2 * ry));
					b[i] -= 2 * ry * dy * heat_flux_C;
				}
			}
			//2. pureright
			else if (m_Boundary_Status == 2)
			{
				if (col == m_dim_num - 1 && row != 0 && row != m_dim_num - 1)
				{
					tripletList.push_back(T(i, i, 1));
				}
				else if (row == 0 && col != 0)
				{
					tripletList.push_back(T(i, i, 1 + 2 * ry));
					tripletList.push_back(T(i, (row + 1) * m_dim_num + col, -2 * ry));
					b[i] -= 2 * ry * dy * heat_flux_C;
				}
				else if (col != 0 && row == m_dim_num - 1)
				{
					tripletList.push_back(T(i, i, 1 + 2 * ry));
					tripletList.push_back(T(i, (row - 1) * m_dim_num + col, -2 * ry));
					b[i] -= 2 * ry * dy * heat_flux_C;
				}	
				else 
				{
					tripletList.push_back(T(i, i, 1 + 2 * rx));
					tripletList.push_back(T(i, row * m_dim_num + (col + 1), -2 * rx));
					b[i] -= 2 * rx * dx * heat_flux_C;
				}
			}
			//3. topleft
			else if (m_Boundary_Status == 3)
			{
				if (row == 0 && col != m_dim_num - 1)
				{
					tripletList.push_back(T(i, i, 1));
				}
				else if (col == 0 && row != m_dim_num - 1)
				{
					tripletList.push_back(T(i, i, 1));
				}
				else if (col == m_dim_num - 1 && row != m_dim_num - 1)
				{
					tripletList.push_back(T(i, i, 1 + 2 * rx));
					tripletList.push_back(T(i, row * m_dim_num + (col - 1), -2 * rx));
					b[i] -= 2 * rx * dx * heat_flux_C;
				}
				else
				{
					tripletList.push_back(T(i, i, 1 + 2 * ry));
					tripletList.push_back(T(i, (row - 1) * m_dim_num + col, -2 * ry));
					b[i] -= 2 * ry * dy * heat_flux_C;
				}
			}
			//4. topright
			else if (m_Boundary_Status == 4)
			{
				if (row == 0 && col != 0)
				{
					tripletList.push_back(T(i, i, 1));
				}
				else if (col == m_dim_num - 1 && row != m_dim_num - 1)
				{
					tripletList.push_back(T(i, i, 1));
				}
				else if (col == 0 && row != m_dim_num - 1)
				{
					tripletList.push_back(T(i, i, 1 + 2 * rx));
					tripletList.push_back(T(i, row * m_dim_num + (col + 1), -2 * rx));
					b[i] -= 2 * rx * dx * heat_flux_C;
				}
				else
				{
					tripletList.push_back(T(i, i, 1 + 2 * ry));
					tripletList.push_back(T(i, (row - 1) * m_dim_num + col, -2 * ry));
					b[i] -= 2 * ry * dy * heat_flux_C;
				}
			}
		}
		//if (col == 0 || row == m_dim_num - 1)
		//{
		//	tripletList.push_back(T(i, i, 1));
		//}
		//else if (row == 0 && col > 0 && col < m_dim_num - 1)
		//{
		//	tripletList.push_back(T(i, i, 1 + 2 * (rx + ry)));
		//	tripletList.push_back(T(i, (row + 1) * m_dim_num + col, -2*ry));
		//	tripletList.push_back(T(i, row * m_dim_num + (col + 1), -rx));
		//	tripletList.push_back(T(i, row * m_dim_num + (col - 1), -rx));
		//}
		//else if (col == m_dim_num - 1 && row > 0 && row < m_dim_num - 1)
		//{
		//	tripletList.push_back(T(i, i, 1 + 2 * (rx + ry)));
		//	tripletList.push_back(T(i, (row + 1) * m_dim_num + col, -ry));
		//	tripletList.push_back(T(i, (row - 1) * m_dim_num + col, -ry));
		//	tripletList.push_back(T(i, row * m_dim_num + (col - 1), -2*rx));
		//}
		//else if (col == m_dim_num - 1 && row == 0)
		//{
		//	tripletList.push_back(T(i, i, 1 + 2 * (rx + ry) * 2 * rx * dx));
		//	tripletList.push_back(T(i, (row + 1) * m_dim_num + col, -2*ry));
		//	tripletList.push_back(T(i, row * m_dim_num + (col - 1), -2*rx));
		//}
		else
		{
			tripletList.push_back(T(i, i, 1 + 2 * (rx + ry)));
			tripletList.push_back(T(i, (row + 1) * m_dim_num + col, -ry));
			tripletList.push_back(T(i, (row - 1) * m_dim_num + col, -ry));
			tripletList.push_back(T(i, row * m_dim_num + (col + 1), -rx));
			tripletList.push_back(T(i, row * m_dim_num + (col - 1), -rx));
		}
	}

	//3. Construct A
	SpMat A(m, m);            // A matrix at the left side of the equation
	A.setFromTriplets(tripletList.begin(), tripletList.end());  // sent the triple coefficient to matrix

	//4. Solve problem:
	Eigen::SparseLU<SpMat> solver(A);
	//Eigen::SimplicialCholesky<SpMat> solver(A);
	solver.compute(A);
	if (solver.info() != Eigen::Success)
	{
		qDebug() << "Compute Matrix is error" ;
		return;
	}
	Eigen::VectorXd x = solver.solve(b);

	//5. update the solved value
	for (int i = 0; i < m; i++)
	{
		if (m_Points[i].ReturnConsFlag() != true)
			m_Points[i].SetTemp(x[i]);
		if (m_Points[i].ReturnDamageHoleflag() && m_Points[i].ReturnTemp() < 25)
			m_Points[i].SetTemp(m_Points[i].ReturnTemp() + 2);
		m_Temp_max = std::max(m_Points[i].ReturnTemp(), m_Temp_max);
		m_Temp_min = std::min(m_Points[i].ReturnTemp(), m_Temp_min);
	}
}

//-----------NO USE ANYMORE------------//
void TempGrid::UpdateHeat()
{
	double dt = pow(m_unit_lenx, 2)/100.0;
	double p_sum;
	for (int i = 0; i < m_Points.size(); i++)
	{
		if (m_Points[i].ReturnConsFlag() != true)
		{
			int row = i / m_dim_num;
			int col = i % m_dim_num;

			if (row == 0) m_Points[i].SetOldTemp(0);
			else if (col == m_dim_num - 1) m_Points[i].SetOldTemp(0);
			else if (row == m_dim_num - 1) m_Points[i].SetOldTemp(0);
			else if (col == 0) m_Points[i].SetOldTemp(0);
			else m_Points[i].SetOldTemp(m_Points[i].ReturnTemp());
		}
	}

	for (int i = 0; i < m_Points.size(); i++)
	{
		if (m_Points[i].ReturnConsFlag() != true)
		{
			double w1 = 1.0 / pow(m_unit_lenx, 2); //x direction1
			double w2 = 1.0 / pow(m_unit_lenx, 2); //x direction2
			double w3 = 1.0 / pow(m_unit_lenx, 2); //y direction1
			double w4 = 1.0 / pow(m_unit_lenx, 2); //y direction2

			//float lambda = Q * d / (A * dT);
			//float Dx 
			double Boltzmann_lambdax = 1;//(2 * D) / ((3 * k * rol) * pow(unit_lenx,2));
			double Boltzmann_lambday = 1;//(2 * D) / ((3 * k * rol) * pow(unit_leny, 2));

			int row = i / m_dim_num;
			int col = i % m_dim_num;

			//if (row == 0)
			//{
			//	if (col != 0 && col != m_dim_num - 1)
			//	{
			//		p_sum += (1.0 / 3.0) * Boltzmann_lambdax * (m_Points[(row + 1) * m_dim_num + col].ReturnTemp() - m_Points[i].ReturnTemp());
			//		p_sum += (1.0 / 3.0) * Boltzmann_lambday * (m_Points[row * m_dim_num + (col - 1)].ReturnTemp() - m_Points[i].ReturnTemp());
			//		p_sum += (1.0 / 3.0) * Boltzmann_lambday * (m_Points[row * m_dim_num + (col + 1)].ReturnTemp() - m_Points[i].ReturnTemp());
			//	}
			//	else if (col == 0)
			//	{
			//		p_sum += (1.0 / 2.0) * Boltzmann_lambdax * (m_Points[(row + 1) * m_dim_num + col].ReturnTemp() - m_Points[i].ReturnTemp());
			//		p_sum += (1.0 / 2.0) * Boltzmann_lambday * (m_Points[row * m_dim_num + (col + 1)].ReturnTemp() - m_Points[i].ReturnTemp());
			//	}
			//	else if (col == m_dim_num - 1)
			//	{
			//		p_sum += (1.0 / 2.0) * Boltzmann_lambdax * (m_Points[(row + 1) * m_dim_num + col].ReturnTemp() - m_Points[i].ReturnTemp());
			//		p_sum += (1.0 / 2.0) * Boltzmann_lambday * (m_Points[row * m_dim_num + (col - 1)].ReturnTemp() - m_Points[i].ReturnTemp());
			//	}
			//
			//}
			//else if (row == m_dim_num - 1)
			//{
			//	if (col != 0 && col != m_dim_num - 1)
			//	{
			//		p_sum += (1.0 / 3.0) * Boltzmann_lambdax * (m_Points[(row - 1) * m_dim_num + col].ReturnTemp() - m_Points[i].ReturnTemp());
			//		p_sum += (1.0 / 3.0) * Boltzmann_lambday * (m_Points[row * m_dim_num + (col - 1)].ReturnTemp() - m_Points[i].ReturnTemp());
			//		p_sum += (1.0 / 3.0) * Boltzmann_lambday * (m_Points[row * m_dim_num + (col + 1)].ReturnTemp() - m_Points[i].ReturnTemp());
			//	}
			//	else if (col == 0)
			//	{
			//		p_sum += (1.0 / 2.0) * Boltzmann_lambdax * (m_Points[(row - 1) * m_dim_num + col].ReturnTemp() - m_Points[i].ReturnTemp());
			//		p_sum += (1.0 / 2.0) * Boltzmann_lambday * (m_Points[row * m_dim_num + (col + 1)].ReturnTemp() - m_Points[i].ReturnTemp());
			//	}
			//	else if (col == m_dim_num - 1)
			//	{
			//		p_sum += (1.0 / 2.0) * Boltzmann_lambdax * (m_Points[(row - 1) * m_dim_num + col].ReturnTemp() - m_Points[i].ReturnTemp());
			//		p_sum += (1.0 / 2.0) * Boltzmann_lambday * (m_Points[row * m_dim_num + (col - 1)].ReturnTemp() - m_Points[i].ReturnTemp());
			//	}
			//}
			if (row == 0) m_Points[i].SetTemp(0);
			else if (col == m_dim_num - 1) m_Points[i].SetTemp(0);	
			else if (row == m_dim_num - 1) m_Points[i].SetTemp(0);
			else if (col == 0) m_Points[i].SetTemp(0);
			else
			{
				p_sum += w1 * Boltzmann_lambdax * (m_Points[(row - 1) * m_dim_num + col].ReturnOldTemp() - m_Points[i].ReturnOldTemp());
				p_sum += w2 * Boltzmann_lambdax * (m_Points[(row + 1) * m_dim_num + col].ReturnOldTemp() - m_Points[i].ReturnOldTemp());
				p_sum += w3 * Boltzmann_lambday * (m_Points[row * m_dim_num + (col + 1)].ReturnOldTemp() - m_Points[i].ReturnOldTemp());
				p_sum += w4 * Boltzmann_lambday * (m_Points[row * m_dim_num + (col + 1)].ReturnOldTemp() - m_Points[i].ReturnOldTemp());
				m_Points[i].SetTemp(m_Points[i].ReturnOldTemp() + dt * p_sum);
			}
			 
			//int k = (row - 1) % m_dim_num * m_dim_num + col;
			//if (k < 0) k = m_Points.size() - 1 + k;
			//p_sum += w1 * Boltzmann_lambdax * (m_Points[k].ReturnOldTemp() - m_Points[i].ReturnOldTemp());
			//p_sum += w2 * Boltzmann_lambdax * (m_Points[(row + 1) % m_dim_num * m_dim_num + col].ReturnOldTemp() - m_Points[i].ReturnOldTemp());
			//k = row * m_dim_num + (col - 1) % m_dim_num;
			//if (k < 0) k = m_dim_num - 1;
			//p_sum += w3 * Boltzmann_lambday * (m_Points[k].ReturnOldTemp() - m_Points[i].ReturnOldTemp());
			//p_sum += w4 * Boltzmann_lambday * (m_Points[row * m_dim_num + (col + 1) % m_dim_num].ReturnOldTemp() - m_Points[i].ReturnOldTemp());
			
			//k = m_Points[i].ReturnTemp() + dt * p_sum;
			//m_Points[i].SetTemp(m_Points[i].ReturnOldTemp() + dt * p_sum);
			m_Temp_max = std::max(m_Points[i].ReturnTemp(), m_Temp_max);
		    m_Temp_min = std::min(m_Points[i].ReturnTemp(), m_Temp_min);
		}
	
	}

}
//-----------NO USE ANYMORE------------//


std::vector<TempGridPoint> TempGrid::ReturnGrid()
{
	return m_Points;
}

std::vector<TempVectorPoint> TempGrid::ReturnVectorGrid()
{
	return m_VectorPoints;
}

//cv::Mat temp_map_gray;
//int g_nAlphaValuesSlider;
//int g_nBetaValuesSlider;
//
//static void on_trackbar(int, void*)
//{
//	cv::Mat out;
//	Canny(temp_map_gray, out, g_nAlphaValuesSlider, 200, 7, true);
//	cv::imwrite("test3.png", out);
//	cv::imshow("Test3", out);
//}

	//cv::namedWindow("canny", 1);
	//cv::createTrackbar("alpha", "canny", &g_nAlphaValuesSlider, 200, on_trackbar);
	//////cv::createTrackbar("beta", "canny", &g_nBetaValuesSlider, 100, on_trackbar);
	////
	//on_trackbar(g_nAlphaValuesSlider, 0);

void DrawRulerColor2(cv::Mat m)
{
	int xmove = 50;
	cv::circle(m, cv::Point2i(19 + xmove, 439), 3, cv::Scalar(0, 0, 0), -1);

	putText(m, std::to_string(0), cv::Point2i(xmove - 10, 439), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2, false);

	cv::Mat colorbar = cv::imread("./CrackMaps/bar2.png");
	cv::resize(colorbar, colorbar, cv::Size(25, 420));
	cv::Mat dst_roi = m(cv::Rect(xmove + 20, 19, colorbar.cols, colorbar.rows));
	colorbar.copyTo(dst_roi);

	for (int y = 19; y < 429; y += 5)
	{
		if ((y + 1) % 40 == 10)
		{
			cv::line(m, cv::Point2i(19 + xmove, y), cv::Point2i(30 + xmove, y), cv::Scalar(0, 0, 0), 1);
			double value = 40 - 3.64 * (float)((y + 1) - 10) / 40;
			std::stringstream stream;
			stream << std::fixed << std::setprecision(2) << value;
			std::string s = stream.str();
			putText(m, s, cv::Point2i(xmove - 30, y + 4), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2, false);
		}
	}
	cv::arrowedLine(m, cv::Point2i(19 + xmove, 439), cv::Point2i(19 + xmove, 19), cv::Scalar(0, 0, 0), 2, 8, 0, 0.015);
	putText(m, "Degrees Celsius", cv::Point2i(xmove - 15, 465), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2, false);
	putText(m, ".C", cv::Point2i(xmove - 0, 490), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2, false);
}

void TempGrid::UpdateTemerature()
{
	cv::Mat src = cv::Mat::zeros(m_dim_num, m_dim_num, CV_8UC3); //color_jet
	cv::Mat src2 = cv::Mat::zeros(m_dim_num, m_dim_num, CV_8UC1); //grey
	cv::Mat src3 = cv::Mat::zeros(m_dim_num, m_dim_num, CV_8UC3); //color_heat
	uchar* pxvec = src.ptr<uchar>(0);
	uchar* pxvec2 = src2.ptr<uchar>(0);
	uchar* pxvec3 = src3.ptr<uchar>(0);

	for (int i = 0; i < src.rows; i++)
	{
		pxvec = src.ptr<uchar>(i);
		pxvec2 = src2.ptr<uchar>(i);
		pxvec3 = src3.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j ++)
		{
			double c = (m_Points[i * m_dim_num + j].ReturnTemp() - 273.15) / abs(40 - 0);

			QVector3D d1 = 255 * color_t_Jet(c);
			pxvec[3 * j + 0] = d1.z();
			pxvec[3 * j + 1] = d1.y();
			pxvec[3 * j + 2] = d1.x();

			pxvec2[j] = c*255;

			QVector3D d2 = 255 * color_t_Heat(c);
			pxvec3[3 * j + 0] = d2.z();
			pxvec3[3 * j + 1] = d2.y();
			pxvec3[3 * j + 2] = d2.x();
		}
	}

	m_TempMap = cv::Mat(m_texturesize_row, m_texturesize_col, CV_8UC3);
	cv::resize(src, m_TempMap, cv::Size(m_texturesize_row, m_texturesize_col), cv::INTER_CUBIC); 
	DrawRulerColor2(m_TempMap);
	cv::imwrite("./CrackMaps/Texture_Temperature.png", m_TempMap);
	QImage img0 = QImage("./CrackMaps/Texture_Temperature.png");
	//Q_ASSERT(!img0.isNull());
	//if (m_Tex_Temperature)
	//	delete m_Tex_Temperature;
	m_Tex_Temperature = new QOpenGLTexture(img0);

	cv::Mat h = cv::Mat(m_texturesize_row, m_texturesize_col, CV_8UC3);
	cv::resize(src3, h, cv::Size(m_texturesize_row, m_texturesize_col), cv::INTER_CUBIC);
	//blur(src3, src3, cv::Size(8, 8));
	cv::imwrite("./CrackMaps/Texture_Temperature_heat.png", h);
	img0 = QImage("./CrackMaps/Texture_Temperature_heat.png");
	//Q_ASSERT(!img0.isNull());
	//if (m_Tex_Temperature)
	//	delete m_Tex_Temperature;
	m_Tex_Temperature_heat = new QOpenGLTexture(img0);


	//cv::Mat temp_map_gray;
	////cvtColor(m_TempMap, temp_map_gray, cv::COLOR_BGR2GRAY);
	//temp_map_gray = cv::Mat(m_texturesize_row, m_texturesize_col, CV_8UC1);
	//cv::resize(src2, temp_map_gray, cv::Size(m_texturesize_row, m_texturesize_col), cv::INTER_CUBIC);
	//cv::imwrite("Texture_Temperature_gray.png", temp_map_gray);
	////cv::imshow("gray", temp_map_gray);
	//
	//
	//blur(temp_map_gray, temp_map_gray, cv::Size(8, 8));
	//cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
	////filter2D(temp_map_gray, temp_map_gray, CV_8UC1, kernel);
	//cv::imwrite("Texture_Temperature_gray_threshold.png", temp_map_gray);
	////cv::imshow("gray_thresh", temp_map_gray);
	//
	//
	//Canny(temp_map_gray, temp_map_gray, 0, 200, 7, true);
	//cv::imwrite("Texture_Contour.png", temp_map_gray);
	////cv::imshow("contour", temp_map_gray);
	//img0 = QImage("Texture_Contour.png");
	////Q_ASSERT(!img0.isNull());
	////if (m_Tex_Contour) 
	////	delete m_Tex_Contour;
	//m_Tex_Contour = new QOpenGLTexture(img0);
}

void TempGrid::UpdateVecterField()
{
	for (int i = 0; i < m_VectorPoints.size(); i++) //row
	{
		QMatrix4x4 w2 = QMatrix4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
		w2.translate(m_VectorPoints[i].ReturnPos().x, m_VectorPoints[i].ReturnPos().y);

		auto k = (CalculateGradient(m_width * m_VectorPoints[i].ReturnPos().x, m_height * m_VectorPoints[i].ReturnPos().y));
		double angle = k.first;
		double scale = log(k.second)/log(5000);

		w2.translate(0, 0, 2 * 0.5); // move to the corresponding layer
		w2.scale( (0.05*30*pow(m_Field_dense+1,-1))*scale,  0.05 * 30 * pow(m_Field_dense + 1, -1) *scale);
		//w2.scale(0.05 , 0.05 );
		w2.rotate(angle, 0.f, 0.f, 1.f);
		m_VectorPoints[i].SetTMatrix(w2);
	}
	m_contour_flag = true;
}

void TempGrid::UpdateContourField()
{
	if (m_contour_flag)
	{
		for (int i = 0; i < m_VectorPoints.size(); i++) //row
		{
			QMatrix4x4 w2 = m_VectorPoints[i].ReturnTMatix();
			w2.rotate(90, 0.f, 0.f, 1.f);
			m_VectorPoints[i].SetTMatrix(w2);
		}
		m_contour_flag = false;
	}
	else
	{
		qWarning() << "Please update the temperature's Gradient Field first.";
	}
}

std::pair<double, double> TempGrid::CalculateGradient(double x, double y)
{
	std::pair<double, double> gradient;
	// xx xx 1 xx xx 
	// xx xx 2 xx xx
	// 3  4  5  6  7
	// xx xx 8 xx xx 
	// xx xx 9 xx xx
	double dt = 0.000005;
	
	if (x == 2.5 || y == 2.5 || x == -2.5 || y == -2.5)
	{
		gradient.first = 0.0;
		gradient.second = 0.0;

		return gradient;
	}
	else
	{
		double q1 = GetTemp(x, y + 2 * dt);
		double q2 = GetTemp(x, y + dt);
		double q8 = GetTemp(x, y - dt);
		double q9 = GetTemp(x, y - 2 * dt);

		double q7 = GetTemp(x + 2 * dt, y);
		double q6 = GetTemp(x + dt, y);
		double q4 = GetTemp(x - dt, y);
		double q3 = GetTemp(x - 2 * dt, y);

		double dfdx = (-q7 + 8 * q6 - 8 * q4 + q3) / (12 * dt);
		double dfdy = (-q1 + 8 * q2 - 8 * q8 + q9) / (12 * dt);

		gradient.first = std::atan2(-dfdy, -dfdx) * 180.0 / 3.1415926;
		gradient.second = sqrt(pow(dfdy, 2) + pow(dfdx, 2));

		return gradient;
	}
}

double TempGrid::GetTemp(double x, double y)
{
	double heat;
	double xc =  x;
	double yc =  y;
	double dx = (m_width / (m_dim_num - 1));
	double dy = (m_height / (m_dim_num - 1));
	int xunit = floor((xc + 0.5 * m_width) / dx); //Q12 ~ X
	int yunit = floor((0.5 * m_height - yc) / dy); //Q12 ~ Y

	double x1 = -0.5 * m_width + xunit * dx;
	double x2 = -0.5 * m_width + (xunit + 1) * dx;
	double y1 = 0.5 * m_height - (yunit + 1) * dy;
	double y2 = 0.5 * m_height - yunit * dy;

	double Q12 = m_Points[xunit + yunit * m_dim_num].ReturnTemp();
	double Q11 = m_Points[xunit + (yunit + 1) * m_dim_num].ReturnTemp();
	double Q21 = m_Points[(xunit + 1) + (yunit + 1) * m_dim_num].ReturnTemp();
	double Q22 = m_Points[(xunit + 1) + yunit * m_dim_num].ReturnTemp();

	heat = ((y2 - yc) / (y2 - y1)) * (((x2 - xc) / (x2 - x1)) * Q11 + ((xc - x1) / (x2 - x1)) * Q21)
		+ ((yc - y1) / (y2 - y1)) * (((x2 - xc) / (x2 - x1)) * Q12 + ((xc - x1) / (x2 - x1)) * Q22);

	return heat;
}

