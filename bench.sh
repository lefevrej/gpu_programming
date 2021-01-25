echo "bin/sobel_gpu JuliaSet.1024.pgm test.pgm 1024 1" > bench.txt
bin/sobel_gpu JuliaSet.1024.pgm test.pgm 1024 1 >> bench.txt
echo "" >> bench.txt
echo "bin/sobel_gpu JuliaSet.1024.pgm test.pgm 512 2" >> bench.txt
bin/sobel_gpu JuliaSet.1024.pgm test.pgm 512 2 >> bench.txt
echo "" >> bench.txt
echo "bin/sobel_gpu JuliaSet.1024.pgm test.pgm 128 4" >> bench.txt
bin/sobel_gpu JuliaSet.1024.pgm test.pgm 128 4 >> bench.txt
echo "" >> bench.txt
echo "bin/sobel_gpu JuliaSet.1024.pgm test.pgm 64 16" >> bench.txt
bin/sobel_gpu JuliaSet.1024.pgm test.pgm 64 16 >> bench.txt
echo "" >> bench.txt
echo "bin/sobel_gpu JuliaSet.1024.pgm test.pgm 32 32" >> bench.txt
bin/sobel_gpu JuliaSet.1024.pgm test.pgm 32 32 >> bench.txt
