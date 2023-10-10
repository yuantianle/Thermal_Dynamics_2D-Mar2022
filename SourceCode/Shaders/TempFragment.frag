#version 330 core
in vec3 vs_position;
in vec3 vs_color;
in vec2 vs_texcoord;

uniform int Erosionlevel;
uniform int Oxidationlevel;
out vec4 fs_color;

uniform sampler2D texture0;

//uniform bool contour_flag;
//uniform bool vector_flag;


mat3 GAUSS_operator(float sigma)
{
	float N = 3;
    mat3 val;
    float R = (N-1)/2;    //***** Use N-1 because the array start from O rather than 1
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            float r = (i-R)*(i-R)+(j-R)*(j-R);
            float res = exp(-r/(2*sigma*sigma));
            res = (res /(2*3.14*sigma*sigma));
            val[i][j] = res;
        }
    }    
	return val;
}

vec4 dip_filter(mat3 _filter, sampler2D _image, vec2 _xy, vec2 texSize)               
{                                                									  
	float r = 2;
	mat3 _filter_pos_delta_x=mat3(vec3(-r, 0.0, r), vec3(-r, 0.0 ,r) ,vec3(-r,0.0,r));          
    mat3 _filter_pos_delta_y=mat3(vec3(-r,-r,-r),vec3(0.0,0.0,0.0),vec3(r,r,r));         
	vec4 final_color = vec4(0.0, 0.0, 0.0, 0.0);                                      
	for(int i = 0; i<3; i++)                                                          
	{                                                                                 
		for(int j = 0; j<3; j++)                                                      
		{                                                                             
			vec2 _xy_new = vec2(_xy.x + _filter_pos_delta_x[i][j], _xy.y + _filter_pos_delta_y[i][j]); 
			vec2 _uv_new = vec2(_xy_new.x/texSize.x, _xy_new.y/texSize.y); 
			vec4 color = texture2D(_image,_uv_new);
			//if (color.a == 0) discard;
			final_color += color * _filter[i][j];            
		}																		
	}																			
	return final_color;															
}

vec4 dip_filter_gauss(mat3 _filter, sampler2D _image, vec2 _xy, vec2 texSize)               
{                        
	float sigma = 1;                                                         //step1 gauss blur & grey
	mat3 _smooth_fil = GAUSS_operator(sigma)*2;                              //

	float r = 2.5;
	mat3 _filter_pos_delta_x=mat3(vec3(-r, 0.0, r), vec3(-r, 0.0 ,r) ,vec3(-r,0.0,r));          
    mat3 _filter_pos_delta_y=mat3(vec3(-r,-r,-r),vec3(0.0,0.0,0.0),vec3(r,r,r));   
	vec4 final_color = vec4(0.0, 0.0, 0.0, 0.0);                                      
	for(int i = 0; i<3; i++)                                                          
	{                                                                                 
		for(int j = 0; j<3; j++)                                                      
		{                                                                             
			vec2 _xy_new = vec2(_xy.x + _filter_pos_delta_x[i][j], _xy.y + _filter_pos_delta_y[i][j]); 
			vec2 _uv_new = vec2(_xy_new.x/texSize.x, _xy_new.y/texSize.y);   

			//final_color += texture2D(_image,_uv_new)* _filter[i][j];
			vec4 gauss_color = dip_filter(_smooth_fil, _image, _xy_new, texSize); //
			final_color += gauss_color * _filter[i][j];                                  //
		}																		
	}																			
	return final_color;															
}	

float sobel_gaussblur_filter(mat3 _filter1, mat3 _filter2, sampler2D _image, vec2 _xy, vec2 texSize)               
{   
	float sigma = 1;                                                         //step1 gauss blur & grey
	mat3 _smooth_fil = GAUSS_operator(sigma)*2;

	float r = 2;
	mat3 _filter_pos_delta_x=mat3(vec3(-r, 0.0, r), vec3(-r, 0.0 ,r) ,vec3(-r,0.0,r));          
    mat3 _filter_pos_delta_y=mat3(vec3(-r,-r,-r),vec3(0.0,0.0,0.0),vec3(r,r,r));   

	vec4 gauss_color = vec4(0.0, 0.0, 0.0, 0.0); 
	float final_color;  
	float final1_color; 
	float final2_color; 
	for(int i = 0; i<3; i++)                                                          
	{                                                                                 
		for(int j = 0; j<3; j++)                                                      
		{                                                                             
			vec2 _xy_new = vec2(_xy.x + _filter_pos_delta_x[i][j], _xy.y + _filter_pos_delta_y[i][j]); 
			vec2 _uv_new = vec2(_xy_new.x/texSize.x, _xy_new.y/texSize.y); 

			vec4 gauss_color = dip_filter_gauss(_smooth_fil, _image, _xy_new, texSize); 
			vec4 W = vec4(0.3,0.45,0.1,0);
			float luminace = dot(gauss_color,W);

			//if (color.r == color.g && color.g == color.b && color.b == 1) discard;
			final1_color += luminace * _filter1[i][j]; 
			final2_color += luminace * _filter2[i][j];
			
		}																		
	}
	
	final_color += sqrt(final1_color * final1_color + final2_color * final2_color);		//gradientMagnitude;
	//vec2 normalizedDirection = normalize(final1_color,final2_color);
	return final_color;															
}	

 void main()
{

     //if(contour_flag == false && vector_flag == false)
	 //	fs_color = texture2D(texture_temper, vs_texcoord);//vec4(vs_color,1.0);
	 //else if (contour_flag == true && vector_flag == false)
	 //{
	 //	vec4 contour = texture2D(texture_contour, vs_texcoord);
	 //	if (contour.x > 0.5) fs_color = vec4(1,1,1,1);
	 //else fs_color = texture2D(texture_temper, vs_texcoord);
	 //}
	 fs_color = texture2D(texture0, vs_texcoord);
}