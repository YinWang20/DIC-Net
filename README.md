# DIC-Net: Upgrade the Performance of Traditional DIC with Hermite Dataset and Convolution Neural Network
Hermite dataset generation implementation (matlab) and DIC-Net CNN implementation (pytorch).

## Introduction
Our method is currently the SOTA of 2D-DeepDIC method in terms of accuracy and spatial resolution(SR).<br>

In the experimental test set with 3216 samples, DIC-Net-d achieves an absolute pixel error (MAE) score of 0.0130 pixels and 0.0126 pixels in the x direction and y direction which is only 48.5% and 47.9% of the prior best method. The spatial resolution (SR) is 17.25 pixels with a noise level of 0.0136, and the metrological performance indicator is 0.234 (lower is better) which outperforms existing traditional and non-traditional methods.

For more details, please refer to our paper: https://doi.org/10.1016/j.optlaseng.2022.107278

* **Frame of DIC-Net**
<img src="Images/architechture.PNG" alt="Architechture" title="title">

* **Frame of Hermite Dataset**
<img src="Images/Hermite Dataset.PNG" alt="Dataset" title="title">

## Main Results
* **Prediction error statistics of three networks on 3216 samples**
<table class="MsoNormalTable" border="0" cellspacing="0" cellpadding="0" width="100%" style="width:100.0%;border-collapse:collapse;mso-yfti-tbllook:1184;mso-padding-alt:
 0cm 5.4pt 0cm 5.4pt">
 <tbody><tr style="mso-yfti-irow:0;mso-yfti-firstrow:yes;height:13.5pt">
  <td width="37%" nowrap="" rowspan="2" style="width:37.38%;border:solid windowtext 1.0pt;
  border-left:none;padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">Displacement mode (number
  of samples)<o:p></o:p></span></p>
  </td>
  <td width="21%" nowrap="" rowspan="2" style="width:21.34%;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">Network<o:p></o:p></span></p>
  </td>
  <td width="20%" nowrap="" colspan="2" style="width:20.62%;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-fareast-theme-font:
  minor-fareast;mso-bidi-font-family:&quot;Times New Roman&quot;">Average </span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">MAE/pixel<o:p></o:p></span></p>
  </td>
  <td width="20%" nowrap="" colspan="2" style="width:20.66%;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-fareast-theme-font:
  minor-fareast;mso-bidi-font-family:&quot;Times New Roman&quot;">Average </span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">RMSE/pixel<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:1;height:13.5pt">
  <td width="10%" nowrap="" style="width:10.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext 1.0pt;mso-border-left-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">u<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">v<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext 1.0pt;mso-border-left-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">u<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.34%;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext 1.0pt;mso-border-left-alt:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">v<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:2;height:13.5pt">
  <td width="37%" nowrap="" rowspan="3" style="width:37.38%;border:none;border-right:
  solid windowtext 1.0pt;mso-border-top-alt:solid windowtext 1.0pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="mso-fareast-font-family:宋体;mso-fareast-theme-font:minor-fareast">Pure
  translations</span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:
  宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">
  (785)<o:p></o:p></span></p>
  </td>
  <td width="21%" nowrap="" style="width:21.34%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="left" style="text-align:left;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">DIC-Net-d <o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.3%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0124</span></b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0115</span></b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0155</span></b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.34%;border:none;mso-border-left-alt:
  solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0147</span></b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:3;height:13.5pt">
  <td width="21%" nowrap="" style="width:21.34%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="left" style="text-align:left;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span class="SpellE"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:
  宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">DisplacementNet</span></span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.3%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0182<b><o:p></o:p></b></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0154<b><o:p></o:p></b></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0197<b><o:p></o:p></b></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.34%;border:none;mso-border-left-alt:
  solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0172<b><o:p></o:p></b></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:4;height:13.5pt">
  <td width="21%" nowrap="" style="width:21.34%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="left" style="text-align:left;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span class="SpellE"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:
  宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">StrainNet</span></span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">-f<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0265<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0255<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0442<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.34%;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0418<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:5;height:13.5pt">
  <td width="37%" nowrap="" rowspan="3" style="width:37.38%;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:none;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="mso-fareast-font-family:宋体;mso-fareast-theme-font:minor-fareast">Gaussian</span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"> (445)<o:p></o:p></span></p>
  </td>
  <td width="21%" nowrap="" style="width:21.34%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="left" style="text-align:left;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">DIC-Net-d<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.3%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0128</span></b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0122</span></b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0161</span></b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.34%;border:none;mso-border-left-alt:
  solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" style="text-indent:0cm;mso-char-indent-count:0;line-height:
  normal;mso-pagination:widow-orphan"><b><span lang="EN-US" style="font-size:
  9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;
  color:black;mso-font-kerning:0pt">0.0156</span></b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;
  color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:6;height:13.5pt">
  <td width="21%" nowrap="" style="width:21.34%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="left" style="text-align:left;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span class="SpellE"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:
  宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">DisplacementNet</span></span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.3%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0171<b><o:p></o:p></b></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0164<b><o:p></o:p></b></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0199<b><o:p></o:p></b></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.34%;border:none;mso-border-left-alt:
  solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" style="text-indent:0cm;mso-char-indent-count:0;line-height:
  normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;
  mso-font-kerning:0pt">0.0194<b><o:p></o:p></b></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:7;height:13.5pt">
  <td width="21%" nowrap="" style="width:21.34%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="left" style="text-align:left;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span class="SpellE"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:
  宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">StrainNet</span></span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">-f<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0256<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0265<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0441<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.34%;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" style="text-indent:0cm;mso-char-indent-count:0;line-height:
  normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;
  mso-font-kerning:0pt">0.0433<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:8;height:13.5pt">
  <td width="37%" nowrap="" rowspan="3" style="width:37.38%;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:none;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="mso-fareast-font-family:宋体;mso-fareast-theme-font:minor-fareast">Periodic
  displacement</span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:
  宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">
  (497)<o:p></o:p></span></p>
  </td>
  <td width="21%" nowrap="" style="width:21.34%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="left" style="text-align:left;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">DIC-Net-d<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.3%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;
  mso-font-kerning:0pt">0.0157</span></b><span lang="EN-US" style="font-size:
  9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;
  color:black;mso-font-kerning:0pt;mso-bidi-font-weight:bold"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;
  mso-font-kerning:0pt">0.0169</span></b><span lang="EN-US" style="font-size:
  9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;
  color:black;mso-font-kerning:0pt;mso-bidi-font-weight:bold"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;
  mso-font-kerning:0pt">0.0201</span></b><span lang="EN-US" style="font-size:
  9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;
  color:black;mso-font-kerning:0pt;mso-bidi-font-weight:bold"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.34%;border:none;mso-border-left-alt:
  solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" style="text-indent:0cm;mso-char-indent-count:0;line-height:
  normal;mso-pagination:widow-orphan"><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0217</span></b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt;mso-bidi-font-weight:bold"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:9;height:13.5pt">
  <td width="21%" nowrap="" style="width:21.34%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="left" style="text-align:left;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span class="SpellE"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:
  宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">DisplacementNet</span></span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.3%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.1191<b style="mso-bidi-font-weight:normal"><o:p></o:p></b></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.1539<b style="mso-bidi-font-weight:normal"><o:p></o:p></b></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.1395<b style="mso-bidi-font-weight:normal"><o:p></o:p></b></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.34%;border:none;mso-border-left-alt:
  solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" style="text-indent:0cm;mso-char-indent-count:0;line-height:
  normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;
  mso-font-kerning:0pt">0.1783<b style="mso-bidi-font-weight:normal"><o:p></o:p></b></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:10;height:13.5pt">
  <td width="21%" nowrap="" style="width:21.34%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="left" style="text-align:left;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span class="SpellE"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:
  宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">StrainNet</span></span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">-f<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0288<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0295<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0467<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.34%;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" style="text-indent:0cm;mso-char-indent-count:0;line-height:
  normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;
  mso-font-kerning:0pt">0.0458<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:11;height:13.5pt">
  <td width="37%" nowrap="" rowspan="3" style="width:37.38%;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:none;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="mso-fareast-font-family:宋体;mso-fareast-theme-font:minor-fareast">Strain
  concentrations</span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:
  宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">
  (418)<o:p></o:p></span></p>
  </td>
  <td width="21%" nowrap="" style="width:21.34%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="left" style="text-align:left;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">DIC-Net-d<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.3%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0137</span></b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0122</span></b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0175</span></b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.34%;border:none;mso-border-left-alt:
  solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" style="text-indent:0cm;mso-char-indent-count:0;line-height:
  normal;mso-pagination:widow-orphan"><b><span lang="EN-US" style="font-size:
  9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;
  color:black;mso-font-kerning:0pt">0.0157</span></b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;
  color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:12;height:13.5pt">
  <td width="21%" nowrap="" style="width:21.34%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="left" style="text-align:left;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span class="SpellE"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:
  宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">DisplacementNet</span></span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.3%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0410<b><o:p></o:p></b></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0252<b><o:p></o:p></b></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0529<b><o:p></o:p></b></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.34%;border:none;mso-border-left-alt:
  solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" style="text-indent:0cm;mso-char-indent-count:0;line-height:
  normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;
  mso-font-kerning:0pt">0.0315<b><o:p></o:p></b></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:13;height:13.5pt">
  <td width="21%" nowrap="" style="width:21.34%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="left" style="text-align:left;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span class="SpellE"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:
  宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">StrainNet</span></span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">-f<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0301<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0280<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0504<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.34%;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" style="text-indent:0cm;mso-char-indent-count:0;line-height:
  normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;
  mso-font-kerning:0pt">0.0457<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:14;height:13.5pt">
  <td width="37%" nowrap="" rowspan="3" style="width:37.38%;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:none;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="mso-fareast-font-family:宋体;mso-fareast-theme-font:minor-fareast">Linear
  displacement</span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:
  宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">
  (357)<o:p></o:p></span></p>
  </td>
  <td width="21%" nowrap="" style="width:21.34%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="left" style="text-align:left;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">DIC-Net-d<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.3%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0120</span></b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0117</span></b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0151</span></b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.34%;border:none;mso-border-left-alt:
  solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" style="text-indent:0cm;mso-char-indent-count:0;line-height:
  normal;mso-pagination:widow-orphan"><b><span lang="EN-US" style="font-size:
  9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;
  color:black;mso-font-kerning:0pt">0.0148</span></b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;
  color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:15;height:13.5pt">
  <td width="21%" nowrap="" style="width:21.34%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="left" style="text-align:left;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span class="SpellE"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:
  宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">DisplacementNet</span></span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.3%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0169<b><o:p></o:p></b></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0125<b><o:p></o:p></b></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0187<b><o:p></o:p></b></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.34%;border:none;mso-border-left-alt:
  solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" style="text-indent:0cm;mso-char-indent-count:0;line-height:
  normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;
  mso-font-kerning:0pt">0.0144<b><o:p></o:p></b></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:16;height:13.5pt">
  <td width="21%" nowrap="" style="width:21.34%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="left" style="text-align:left;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span class="SpellE"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:
  宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">StrainNet</span></span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">-f<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0254<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0251<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0435<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.34%;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" style="text-indent:0cm;mso-char-indent-count:0;line-height:
  normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;
  mso-font-kerning:0pt">0.0408<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:17;height:13.5pt">
  <td width="37%" nowrap="" rowspan="3" style="width:37.38%;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:none;border-right:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="mso-fareast-font-family:宋体;mso-fareast-theme-font:minor-fareast">Quadratic
  displacement</span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:
  宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">
  (714)<o:p></o:p></span></p>
  </td>
  <td width="21%" nowrap="" style="width:21.34%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="left" style="text-align:left;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">DIC-Net-d<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.3%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;
  mso-font-kerning:0pt">0.0121<span style="mso-bidi-font-weight:bold"><o:p></o:p></span></span></b></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;
  mso-font-kerning:0pt">0.0117</span></b><span lang="EN-US" style="font-size:
  9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;
  color:black;mso-font-kerning:0pt;mso-bidi-font-weight:bold"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;
  mso-font-kerning:0pt">0.0153</span></b><span lang="EN-US" style="font-size:
  9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;
  color:black;mso-font-kerning:0pt;mso-bidi-font-weight:bold"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.34%;border:none;mso-border-left-alt:
  solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" style="text-indent:0cm;mso-char-indent-count:0;line-height:
  normal;mso-pagination:widow-orphan"><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0149</span></b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt;mso-bidi-font-weight:bold"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:18;height:13.5pt">
  <td width="21%" nowrap="" style="width:21.34%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="left" style="text-align:left;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span class="SpellE"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:
  宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">DisplacementNet</span></span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.3%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0143<b style="mso-bidi-font-weight:normal"><o:p></o:p></b></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0152<b style="mso-bidi-font-weight:normal"><o:p></o:p></b></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0165<b style="mso-bidi-font-weight:normal"><o:p></o:p></b></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.34%;border:none;mso-border-left-alt:
  solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" style="text-indent:0cm;mso-char-indent-count:0;line-height:
  normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;
  mso-font-kerning:0pt">0.0174<b style="mso-bidi-font-weight:normal"><o:p></o:p></b></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:19;height:13.5pt">
  <td width="21%" nowrap="" style="width:21.34%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="left" style="text-align:left;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span class="SpellE"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:
  宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">StrainNet</span></span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">-f<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0250<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0243<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">0.0427<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.34%;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" style="text-indent:0cm;mso-char-indent-count:0;line-height:
  normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;
  mso-font-kerning:0pt">0.0405<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:20;height:13.5pt">
  <td width="37%" rowspan="3" style="width:37.38%;border:solid windowtext 1.0pt;
  border-left:none;padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">Total (3216)<o:p></o:p></span></p>
  </td>
  <td width="21%" nowrap="" style="width:21.34%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="left" style="text-align:left;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">DIC-Net-d<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.3%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-fareast-theme-font:minor-fareast;mso-bidi-font-family:
  &quot;Times New Roman&quot;">0.0130</span></b><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;
  mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-fareast-theme-font:minor-fareast;mso-bidi-font-family:
  &quot;Times New Roman&quot;">0.0126</span></b><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;
  mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-fareast-theme-font:minor-fareast;mso-bidi-font-family:
  &quot;Times New Roman&quot;">0.0165</span></b><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;
  mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.34%;border:none;mso-border-left-alt:
  solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" style="text-indent:0cm;mso-char-indent-count:0;line-height:
  normal;mso-pagination:widow-orphan"><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-fareast-theme-font:
  minor-fareast;mso-bidi-font-family:&quot;Times New Roman&quot;">0.0161</span></b><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:21;height:13.5pt">
  <td width="21%" nowrap="" style="width:21.34%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="left" style="text-align:left;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span class="SpellE"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:
  宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">DisplacementNet</span></span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.3%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-fareast-theme-font:
  minor-fareast;mso-bidi-font-family:&quot;Times New Roman&quot;">0.0356</span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-fareast-theme-font:
  minor-fareast;mso-bidi-font-family:&quot;Times New Roman&quot;">0.0378</span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-fareast-theme-font:
  minor-fareast;mso-bidi-font-family:&quot;Times New Roman&quot;">0.0417</span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.34%;border:none;mso-border-left-alt:
  solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:13.5pt">
  <p class="MsoNormal" style="text-indent:0cm;mso-char-indent-count:0;line-height:
  normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-fareast-theme-font:minor-fareast;mso-bidi-font-family:
  &quot;Times New Roman&quot;">0.0440</span><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;
  mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:22;mso-yfti-lastrow:yes;height:13.5pt">
  <td width="21%" nowrap="" style="width:21.34%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="left" style="text-align:left;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span class="SpellE"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:
  宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">StrainNet</span></span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt">-f<o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.3%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-fareast-theme-font:
  minor-fareast;mso-bidi-font-family:&quot;Times New Roman&quot;">0.0268</span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-fareast-theme-font:
  minor-fareast;mso-bidi-font-family:&quot;Times New Roman&quot;">0.0263</span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.32%;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0;line-height:normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-fareast-theme-font:
  minor-fareast;mso-bidi-font-family:&quot;Times New Roman&quot;">0.0450</span><span lang="EN-US" style="font-size:9.0pt;mso-fareast-font-family:宋体;mso-bidi-font-family:
  &quot;Times New Roman&quot;;color:black;mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
  <td width="10%" nowrap="" style="width:10.34%;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:13.5pt">
  <p class="MsoNormal" style="text-indent:0cm;mso-char-indent-count:0;line-height:
  normal;mso-pagination:widow-orphan"><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-fareast-theme-font:minor-fareast;mso-bidi-font-family:
  &quot;Times New Roman&quot;">0.0427</span><span lang="EN-US" style="font-size:9.0pt;
  mso-fareast-font-family:宋体;mso-bidi-font-family:&quot;Times New Roman&quot;;color:black;
  mso-font-kerning:0pt"><o:p></o:p></span></p>
  </td>
 </tr>
</tbody></table>

* **The performance of spatial resolution in Star5 image set**
<table class="MsoTableGrid" border="1" cellspacing="0" cellpadding="0" width="78%" style="width:78.16%;border-collapse:collapse;border:none;mso-border-top-alt:
 solid windowtext 1.0pt;mso-border-bottom-alt:solid windowtext 1.0pt;
 mso-yfti-tbllook:1184;mso-padding-alt:0cm 5.4pt 0cm 5.4pt;mso-border-insideh:
 none;mso-border-insidev:none">
 <tbody><tr style="mso-yfti-irow:0;mso-yfti-firstrow:yes">
  <td width="41%" valign="top" style="width:41.04%;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:21.0pt"><span lang="EN-US" style="mso-fareast-font-family:宋体;mso-fareast-theme-font:minor-fareast;
  mso-bidi-font-family:&quot;Times New Roman&quot;">Network<o:p></o:p></span></p>
  </td>
  <td width="18%" valign="top" style="width:18.7%;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span lang="EN-US" style="mso-fareast-font-family:
  宋体;mso-fareast-theme-font:minor-fareast;mso-bidi-font-family:&quot;Times New Roman&quot;">SR<o:p></o:p></span></p>
  </td>
  <td width="21%" valign="top" style="width:21.58%;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span lang="EN-US" style="mso-fareast-font-family:
  宋体;mso-fareast-theme-font:minor-fareast;mso-bidi-font-family:&quot;Times New Roman&quot;">Noise
  level<o:p></o:p></span></p>
  </td>
  <td width="18%" valign="top" style="width:18.68%;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span lang="EN-US" style="font-family:&quot;Cambria Math&quot;,&quot;serif&quot;;
  mso-fareast-font-family:宋体;mso-fareast-theme-font:minor-fareast;mso-bidi-font-family:
  &quot;Cambria Math&quot;">𝛼</span><span lang="EN-US" style="mso-fareast-font-family:
  宋体;mso-fareast-theme-font:minor-fareast;mso-bidi-font-family:&quot;Times New Roman&quot;"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:1">
  <td width="41%" valign="top" style="width:41.04%;border:none;mso-border-top-alt:
  solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:21.0pt"><span lang="EN-US" style="mso-fareast-font-family:宋体;mso-fareast-theme-font:minor-fareast;
  mso-bidi-font-family:&quot;Times New Roman&quot;">DIC-Net-d<o:p></o:p></span></p>
  </td>
  <td width="18%" valign="top" style="width:18.7%;border:none;mso-border-top-alt:
  solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="mso-fareast-font-family:宋体;mso-fareast-theme-font:minor-fareast;
  mso-bidi-font-family:&quot;Times New Roman&quot;">17.25<o:p></o:p></span></b></p>
  </td>
  <td width="21%" valign="top" style="width:21.58%;border:none;mso-border-top-alt:
  solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span lang="EN-US" style="mso-fareast-font-family:
  宋体;mso-fareast-theme-font:minor-fareast;mso-bidi-font-family:&quot;Times New Roman&quot;">0.0136<o:p></o:p></span></p>
  </td>
  <td width="18%" valign="top" style="width:18.68%;border:none;mso-border-top-alt:
  solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span lang="EN-US" style="mso-fareast-font-family:
  宋体;mso-fareast-theme-font:minor-fareast;mso-bidi-font-family:&quot;Times New Roman&quot;">0.234<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:2">
  <td width="41%" valign="top" style="width:41.04%;border:none;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:21.0pt"><span class="SpellE"><span lang="EN-US" style="mso-fareast-font-family:宋体;mso-fareast-theme-font:
  minor-fareast;mso-bidi-font-family:&quot;Times New Roman&quot;">StrainNet</span></span><span lang="EN-US" style="mso-fareast-font-family:宋体;mso-fareast-theme-font:minor-fareast;
  mso-bidi-font-family:&quot;Times New Roman&quot;">-f<o:p></o:p></span></p>
  </td>
  <td width="18%" valign="top" style="width:18.7%;border:none;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span lang="EN-US" style="mso-fareast-font-family:
  宋体;mso-fareast-theme-font:minor-fareast">26.93<o:p></o:p></span></p>
  </td>
  <td width="21%" valign="top" style="width:21.58%;border:none;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span lang="EN-US" style="mso-fareast-font-family:
  宋体;mso-fareast-theme-font:minor-fareast">0.0163<o:p></o:p></span></p>
  </td>
  <td width="18%" valign="top" style="width:18.68%;border:none;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span lang="EN-US" style="mso-fareast-font-family:
  宋体;mso-fareast-theme-font:minor-fareast">0.439<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:3;mso-yfti-lastrow:yes">
  <td width="41%" valign="top" style="width:41.04%;border:none;border-bottom:
  solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:21.0pt"><span class="SpellE"><span lang="EN-US" style="mso-fareast-font-family:宋体;mso-fareast-theme-font:
  minor-fareast;mso-bidi-font-family:&quot;Times New Roman&quot;">DisplacemeNet</span></span><span lang="EN-US" style="mso-fareast-font-family:宋体;mso-fareast-theme-font:minor-fareast;
  mso-bidi-font-family:&quot;Times New Roman&quot;"><o:p></o:p></span></p>
  </td>
  <td width="18%" valign="top" style="width:18.7%;border:none;border-bottom:solid windowtext 1.0pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span lang="EN-US" style="mso-fareast-font-family:
  宋体;mso-fareast-theme-font:minor-fareast">52.66<o:p></o:p></span></p>
  </td>
  <td width="21%" valign="top" style="width:21.58%;border:none;border-bottom:
  solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="mso-fareast-font-family:宋体;mso-fareast-theme-font:minor-fareast">0.0041<o:p></o:p></span></b></p>
  </td>
  <td width="18%" valign="top" style="width:18.68%;border:none;border-bottom:
  solid windowtext 1.0pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><b style="mso-bidi-font-weight:normal"><span lang="EN-US" style="mso-fareast-font-family:宋体;mso-fareast-theme-font:minor-fareast">0.216<o:p></o:p></span></b></p>
  </td>
 </tr>
</tbody></table>

## Demo predicted by DIC-Net
### Displacement prediction results compared with other two models
* **Star5 displacement and noise level predicted by three networks.**
<img src="Images/demo1.PNG" alt="demo" title="demo">

* **Tensile plate with hole predicted by three networks.**
<img src="Images/demo2.PNG" alt="demo" title="demo">

### Strain prediction result
* **Star6 strain predicted by DIC-Net-s**
<img src="Images/demo3.PNG" alt="demo" title="demo">

## Our Environment 
### Matalb
* R2020b
* MATLAB Support for MinGW-w64 C/C++ Compiler
### Python
* python 3.0
### CUDA 11.5
### Pytorch 1.10
### Numpy 1.20.3


## Citation
@article{WANG2023107278,<br>
title = {DIC-Net: Upgrade the performance of traditional DIC with Hermite dataset and convolution neural network},<br>
journal = {Optics and Lasers in Engineering},<br>
volume = {160},<br>
pages = {107278},<br>
year = {2023},<br>
issn = {0143-8166},<br>
doi = {https://doi.org/10.1016/j.optlaseng.2022.107278},<br>
url = {https://www.sciencedirect.com/science/article/pii/S0143816622003311},<br>
author = {Yin Wang and Jiaqing Zhao}<br>
}

We're well underway to get the code in order, if you need any help, don't hesitate to contact me (yin-wang20@mails.tsinghua.edu.cn) or the corresponding author (jqzhao@mail.tsinghua.edu.cn) right away!
