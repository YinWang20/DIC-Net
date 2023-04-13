
#ifdef __cplusplus
extern "C"
{
#endif



typedef  double TYPEBSPCOEFF;


 __declspec(dllexport) int	  	SamplesToCoefficients
				(
					TYPEBSPCOEFF	*Image,		/* in-place processing */
					int	Width,		/* width of the image */
					int	Height,		/* height of the image */
					int	SplineDegree/* degree of the spline model */
				);


  __declspec(dllexport) void interpfull(
	 TYPEBSPCOEFF	*pDefInterpImage,	/* input B-spline array of coefficients */
	 int	BSPWidth,		/* width of the image */
	 int	BSPHeight,		/* height of the image */
	 double	*xd,			/* x coordinates where to interpolate */
	 double	*yd,			/* y coordinates where to interpolate */
	 double  *intval, /* interpolated valued,retured */
	 int  NPT,  /* length of xs, intval or ys  */
	 int	InterpolationDegree/* degree of the spline model */
	 );


#ifdef __cplusplus
}
#endif