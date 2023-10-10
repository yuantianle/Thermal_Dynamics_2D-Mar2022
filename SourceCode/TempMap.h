#pragma once
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions>

#include "Primitives.h"

class TempMap
{
private:
	unsigned m_NrOfVertices;
	unsigned m_NrOfIndices;
	GLuint* m_Indices;

	float m_MeshxRot;
	float m_MeshyRot;
	float m_MeshzRot;
	float m_Meshx;
	float m_MeshSx, m_MeshSy;  //scale, we dont use y
	float m_Meshy, m_Meshz;
	GridMesh* m_primitive;
	bool m_VisibleFlag = false;

	QMatrix4x4 m_Transformation;
	GLuint m_CVao;
	GLuint m_CVbo;
	QOpenGLExtraFunctions* m_f;

	QOpenGLTexture* m_Texture_temp;
	QOpenGLTexture* m_Texture_contour;
	QOpenGLTexture* m_Texture_vector;

	bool m_contourflag = false;
	bool m_vectorflag = false;

	void InitVAO(GridMesh* primitive, QOpenGLExtraFunctions* f)
	{
		m_f = f;
		m_primitive = primitive;
		//Set variables
		this->m_NrOfVertices = primitive->GetNrOfTVertices();
		this->m_NrOfIndices = primitive->GetNrOfIndices();
		this->m_Indices = primitive->GetIndices();
		//int t = *(m_Indices+2);
		//VAO, VBO: create boxes, which used to hold bunches of data
		//1.Generate VAO
		f->glGenVertexArrays(1, &this->m_CVao);
		//2.Bind VAO: activate the VAO and anything we do with any other buffers will bind to this VAO
		f->glBindVertexArray(this->m_CVao);
		//3.Generate VBO
		f->glGenBuffers(1, &this->m_CVbo); //Generate a ID for the buffer
		//4.Bind VBO: store array data into the box
		f->glBindBuffer(GL_ARRAY_BUFFER, this->m_CVbo);  //Store array of floats into one part of the box(VBO)
		//5.Send data to VBO
		f->glBufferData(GL_ARRAY_BUFFER, this->m_NrOfVertices * sizeof(TempVertex), primitive->GetTVertices(), GL_STATIC_DRAW); //Send the data in VBO box to the graphics card; GL_STATIC_DRAW: how often u wanna modify the data

		//6.Set vertex attribute pointers and enable them (input assembly): to clearify the attribute meaning of the vertex array

		//position
		f->glEnableVertexAttribArray(0);
		f->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(TempVertex), (GLvoid*)offsetof(TempVertex, position));
		//color
		f->glEnableVertexAttribArray(1);
		f->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(TempVertex), (GLvoid*)offsetof(TempVertex, color));
		//texcoord
		f->glEnableVertexAttribArray(2);
		f->glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(TempVertex), (GLvoid*)offsetof(TempVertex, texcoord));
		////normal
		//f->glEnableVertexAttribArray(3);
		//f->glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, normal));

		//m_Vbo.release();
		//m_Vao.release();
		f->glBindBuffer(GL_ARRAY_BUFFER, 0);
		f->glBindVertexArray(0);
		//f->glEnableVertexAttribArray(0);
	}

	void UpdateUniforms(QOpenGLShaderProgram* shader)
	{
		int m_modelMatrixLoc = shader->uniformLocation("ModelMatrix");
		shader->setUniformValue(m_modelMatrixLoc, m_Transformation);

	}

	void UpdateModelMatrix(QMatrix4x4 uniformmatrix)
	{
		m_Transformation = QMatrix4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
		m_Transformation *= uniformmatrix;
		m_Transformation.translate(m_Meshx, m_Meshy, m_Meshz);
		m_Transformation.rotate(m_MeshxRot, 1.f, 0.f, 0.f);
		m_Transformation.rotate(m_MeshyRot, 0.f, 1.f, 0.f);
		m_Transformation.rotate(m_MeshzRot, 0.f, 0.f, 1.f);
		m_Transformation.scale(m_MeshSx / m_MeshSy, m_MeshSx / m_MeshSy);
	}

public:
	bool m_Curveflag;
	bool m_Anchorflag;
	bool m_Editflag;

public:
	TempMap():m_Texture_temp(0), m_Texture_contour(0)
	{

	}

	~TempMap()
	{
		//m_Vbo.destroy();
		//m_Vao.destroy();
	}

	void SetVisibleFlag(bool b)
	{
		m_VisibleFlag = b;
	}

	bool ReturnVisibleFlag()  //for using the deformation mode
	{
		return m_VisibleFlag;
	}

	void Control(float xmove, float ymove)   //for control botton
	{
		m_Meshx += xmove;
		m_Meshy += ymove;
	}

	void Setting(GridMesh* primitive, QOpenGLExtraFunctions* f)
	{
		//m_Vbo.destroy();
		//m_Vao.destroy();
		//f->glBindBuffer(GL_ARRAY_BUFFER, 0);
		//f->glBindVertexArray(0);
		this->InitVAO(primitive, f);
	}

	void SetTextureTemp(QOpenGLTexture* Texture_temp)
	{
		m_Texture_temp = Texture_temp;
	}

	void SetTextureContour(QOpenGLTexture* Texture_contour)
	{
		m_Texture_contour = Texture_contour;
	}

	void SetContourFlag()
	{
		m_contourflag = !m_contourflag;
	}

	void SetVectorFlag()
	{
		m_vectorflag = !m_vectorflag;
	}


	void SetTextureVector(QOpenGLTexture* Texture_vector)
	{
		m_Texture_vector = Texture_vector;
	}

	void Setting(                      //for situation without layers rendering
		float xRot,
		float yRot,
		float zRot,
		float x,
		float sx,
		float sy,  //scale, we dont use y
		float y,
		float z)
	{
		this->m_MeshxRot = xRot;
		this->m_MeshyRot = yRot;
		this->m_MeshzRot = zRot;
		this->m_Meshx = x;
		this->m_MeshSx = sx;
		this->m_MeshSy = sy;
		this->m_Meshy = y;
		this->m_Meshz = z;
		m_Texture_temp = NULL;
		m_Texture_contour = NULL;
		m_Texture_vector = NULL;
	}	 

	void Render(QOpenGLShaderProgram* shader, QMatrix4x4 m_whole)                                                           //Placed in while
	{
		//m_Vao.bind();
		//m_Vbo.bind();
		InitVAO(m_primitive, m_f);
		m_f->glBindVertexArray(this->m_CVao);
		//m_f->glBindBuffer(GL_ARRAY_BUFFER, this->m_CVbo);

		m_f->glActiveTexture(GL_TEXTURE0);
		shader->setUniformValue("texture_temper", GL_TEXTURE0);
		if (m_Texture_temp != NULL) m_Texture_temp->bind();
		
		
		m_f->glActiveTexture(GL_TEXTURE1);
		shader->setUniformValue("texture_contour", GL_TEXTURE1 - GL_TEXTURE0);
		if (m_Texture_contour != NULL) m_Texture_contour->bind();
		
		
		//m_f->glActiveTexture(GL_TEXTURE2);
		//if (m_Texture_vector != NULL) m_Texture_vector->bind();

		shader->setUniformValue("contour_flag", m_contourflag);
		shader->setUniformValue("vector_flag", m_vectorflag);

		//shader->bind();                   //PREPARE FOR VERTEXARRAY VAO
		this->UpdateModelMatrix(m_whole);
		this->UpdateUniforms(shader);

		//Render
		//8.3.7 DRAW A NEW FRAME
		//glDrawArrays(GL_LINE_STRIP, 0, 8);
		//glDrawElements(GL_TRIANGLES, this->nrOfIndices, GL_UNSIGNED_INT, 0);            //PLACE START DRAWING!

		if (m_VisibleFlag)
		{
			glDrawElements(GL_TRIANGLES, this->m_NrOfIndices, GL_UNSIGNED_INT, m_Indices);
		}
		//m_Vbo.release();
		//shader->release();
		//m_f->glBindVertexArray(0);
	}

};

