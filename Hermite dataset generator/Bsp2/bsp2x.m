function  bsp2x()
% unload bspfull.dll

BSPFULL = ['bspfull' num2str(getmatlabversion())];


if libisloaded(BSPFULL)
  unloadlibrary(BSPFULL);
  disp([ BSPFULL '.dll is unloaded successfully.']);
else
  disp([ BSPFULL '.dll has never been unloaded!']);
end
