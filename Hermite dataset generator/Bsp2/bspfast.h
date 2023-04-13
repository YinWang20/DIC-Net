#ifdef __cplusplus
extern "C"
{
#endif

 double	  InterpolatedValue
				(
					float	*Bcoeff,	/* input B-spline array of coefficients */
					long	Width,		/* width of the image */
					long	Height,		/* height of the image */
					double	x,			/* x coordinate where to interpolate,左上角为0 */
					double	y,			/* y coordinate where to interpolate,左上角为0  */
					long	SplineDegree/* degree of the spline model */
				);

  int	  	SamplesToCoefficients
				(
					float	*Image,		/* in-place processing */
					long	Width,		/* width of the image */
					long	Height,		/* height of the image */
					long	SplineDegree/* degree of the spline model */
				);
 int			InterpolatedValueVect
				(
					float	*Bcoeff,	/* input B-spline array of coefficients */
					long	Width,		/* width of the image */
					long	Height,		/* height of the image */
					double	*xs,			/* x coordinates where to interpolate ,左上角为0 */
					double	*ys,			/* y coordinates where to interpolate ,左上角为0 */
					double  *intval, /* interpolated valued,retured */
					long  nxycount,  /* length of xs, intval or ys  */
					long	SplineDegree/* degree of the spline model */
				);

#ifdef __cplusplus
}
#endif