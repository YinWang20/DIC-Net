function   matlabx32x64 = getmatlabversion()
matlabx32x64 = 32;
a = version('-java');
a = lower(a);
if  ~isempty(strfind(a, '64-bit'))
    matlabx32x64 = 64;
end

%Java 1.6.0_17-b04 with Sun Microsystems Inc. Java HotSpot(TM) 64-Bit Server VM mixed mode
