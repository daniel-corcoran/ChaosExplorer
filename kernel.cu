//---CUDA BUILD---//

#include <iostream>
#include <SFML/Graphics.hpp>
#include <complex>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuComplex.h>
#include <chrono>
#include <thread>
#include <cmath>
//#include <boost/multiprecision/double.hpp>

#define benchmark false

//#include <boost/multiprecision/double.hpp>
#define zoom_multiplier 4
//typedef __double double_t;
//using namespace boost::multiprecision;
class settings {
	double xMax, xMin, cMax, cMin;	//S is -2
	int precision;
	int zoom_incremental = 5;
	double resolution_multiplier; //Reccomended is 20
	double penSize, autozoom_x, autozoom_c;
	bool autozoom;
public:
	settings() {
		if (benchmark==false) {
			xMax = 2;
			xMin = -2;
			cMax = 2;
			cMin = -2;
			std::cout << "Precision? Recommended: 30\n";
			std::cin >> precision;
			std::cout << "\nPixel size? Recommended: 1\n";
			std::cin >> resolution_multiplier;



			std::cout << "zoom resolution incremental: (recommended 2-10)\n";
			std::cin >> zoom_incremental;

			std::cout << "\nAutozoom? (true, false)\n";
			std::cin >> autozoom;
			if (autozoom == true) {
				autozoom_x = -0.7336438924199521;
				autozoom_c = 0.2455211406714035;
			}
		}
		else {
			xMax = 2;
			xMin = -2;
			cMax = 2;
			cMin = -2;
			precision = 100;
			zoom_incremental = 9;
			resolution_multiplier = 1;
			autozoom_x = -0.7336438924199521;
			autozoom_c = 0.2455211406714035;


		}


	}
	bool get_autozoom() {
		return autozoom;
	}
	int get_zoomIncremental(){
		return zoom_incremental; }
	double getAZ_x() {
		return autozoom_x; 
	}
	double getAZ_y() {
		return autozoom_c;
	}
	double getMin_x()
	{
		return xMin;
	}
	double getMin_c() {
		return cMin;
	}
	double getMax_x() {
		return xMax;
	}
	double getMax_c() {
		return cMax;
	}
	double XstepSize() {
		//Screen is 1920 pixels wide. We have a image of width (100 * (xmax - xmin)), so our pixels 
		return  ((xMax - xMin) / 1920) * resolution_multiplier;
	}
	double YstepSize() {
		return ((cMax - cMin) / 1080) * resolution_multiplier;
	}
	int getPrecision() {
		return precision;
	}
	
	//void zoom(double x, double y, sf::View &myView) {
	//	myView.setCenter(sf::Vector2f(x, y));
	//	myView.setSize(myView.getSize().x / 4, myView.getSize().y / 4);
	//	xMin = myView.getCenter().x - (myView.getSize().x / 4);
	//	xMax = myView.getCenter().x + (myView.getSize().x / 4);
	//	cMin = myView.getCenter().y - (myView.getSize().y / 4);
	//	cMax = myView.getCenter().y + (myView.getSize().y / 4);
	//
//	}
	void setPrecision(int newPrecision) {
		precision = newPrecision;
	}
	int getRes() { return resolution_multiplier; }
	void zoom( double x, double y) {
		double centerX = x;
		double centerC = y;
		double sizeX = (xMax - xMin);
		double sizeC = (cMax - cMin);
		xMin = centerX - (sizeX / zoom_multiplier);
		xMax = centerX + (sizeX / zoom_multiplier);
		cMin = centerC - (sizeC / zoom_multiplier);
		cMax = centerC + (sizeC / zoom_multiplier);
	}


};
__global__
void process(int n, int iterations, double Minx, double Miny,  sf::Uint8 *pixels, double x_step, double y_step, int displayX, int displayY) {
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if (id < n) {
		float i;
		int x_index = id % displayX;
		int y_index = ((id - x_index) / displayX);
		//convert x_index to coord_pos
		double c_r = Minx + (x_index * x_step);
		double c_i = Miny + (y_index * y_step);

		double zr = 0; double zi = 0; double zrsqr = 0; double zisqr = 0;

		for (i = 0; i < iterations; i+=2) {
			if (zrsqr + zisqr > 4) {
				break;
			}
			zi = zr * zi;
			zi += zi;
			zi += c_i;
			zr = zrsqr - zisqr + c_r;
			zrsqr = zr * zr;
			zisqr = zi * zi;
			zi = zr * zi;
			zi += zi;
			zi += c_i;
			zr = zrsqr - zisqr + c_r;
			zrsqr = zr * zr;
			zisqr = zi * zi;
			

		

		}
		if ( i >= iterations ) {
			pixels[id * 4] = 30;
			pixels[4 * id + 1] = 10;
			pixels[4 * id + 2] = 60;
			pixels[4 * id + 3] = 255;
		}else {
			
			
			pixels[id * 4] = 255 * (i / iterations);
			pixels[4 * id + 1] = 100 * (i / iterations);
			pixels[4 * id + 2] = 255;
			pixels[4 * id + 3] = 255;
		}
	}
}

int main()
{

	settings instance;



	sf::RenderWindow window(sf::VideoMode(1920, 1080), "Chaos Explorer");

	int displayX = ceil(1920 / instance.getRes());
	int displayY = ceil(1080 / instance.getRes());
	sf::Uint8 *pixels = new sf::Uint8[displayX * displayY * 4];
	sf::Sprite sprite;
	int total_size = displayX * displayY; //How many elements in this array?
	std::cout << "arrays purged succesfully";

	cudaMallocManaged(&pixels, 4*total_size * sizeof(sf::Uint8));
	sf::Texture texture;
	if (benchmark == true) {
		//Run simulation (zoom to pt) ten times and time the duration between it, return time in milliseconds
		sf::Image image;
			while (window.isOpen()) {

				//std::cout << "total length: " << total_size << std::endl;


				{							
					auto start = std::chrono::high_resolution_clock::now(); //Begin the benchmark


					for (int i = 0; i < 10; i++) { //Zoom ten times
						
				
						int iterations = instance.getPrecision();
						double x_step = instance.XstepSize();
						double y_step = instance.YstepSize();
						
						process << <total_size, 1 >> > (total_size, iterations, instance.getMin_x(), instance.getMin_c(), pixels, x_step, y_step, displayX, displayY);
						instance.zoom(instance.getAZ_x(), instance.getAZ_y());
						instance.setPrecision(instance.getPrecision() + instance.get_zoomIncremental());
						cudaDeviceSynchronize();
						
					
						image.create(displayX, displayY, pixels);
						texture.loadFromImage(image);
						sprite.setTexture(texture);
						window.draw(sprite);
						window.display();
						
						
					}
					


					auto stop = std::chrono::high_resolution_clock::now();
					auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

					std::cout << "Score: " << duration.count() << " seconds" << std::endl;

					return 0;
				}


			}

		


	}
	else {
		while (window.isOpen()) {

			std::cout << "total length: " << total_size << std::endl;

		

			{
				//			__double myfloat;
				double Minx = instance.getMin_x();
				double Miny = instance.getMin_c();
				int iterations = instance.getPrecision();
				double x_step = instance.XstepSize();
				double y_step = instance.YstepSize();
				std::cout << "--Process Initiated--\nBetween X: " << Minx << ", " << instance.getMax_x() << "\nC: " << Miny << ", " << instance.getMax_c() << std::endl;
				std::cout << "Processing data points: sending elements to CUDA\n";

				auto start = std::chrono::high_resolution_clock::now();


				process << <total_size, 1 >> > (total_size, iterations, Minx, Miny,pixels, x_step, y_step, displayX, displayY);
				cudaDeviceSynchronize();
				auto stop = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
				std::cout << "Iterative processing time (GPU): " << duration.count() / pow(10, 6) << " seconds" << std::endl;
				start = std::chrono::high_resolution_clock::now();
				std::cout << "creating image from data array...";
				sf::Image image;
				image.create(displayX, displayY, pixels);
				texture.loadFromImage(image);
				sprite.setTexture(texture);
				window.draw(sprite);
				window.display();
				stop = std::chrono::high_resolution_clock::now();
				duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

				std::cout << "Image processing time (CPU): " << duration.count() / pow(10, 6) << " seconds" << std::endl;






				//std::cout << "system 1: \n";


				sf::Event event;


				std::cout << "render complete" << std::endl;
				bool wait = false;

				if (instance.get_autozoom() == false) {
					wait = true;
				}
				else {
					//autozoom routine;
					instance.zoom(instance.getAZ_x(), instance.getAZ_y());
					instance.setPrecision(instance.getPrecision() + 9);

				}
				while (wait) {

					while (window.pollEvent(event))
					{
						if (event.type == sf::Event::Closed) {
							//free(fz);
							//free(pixels);
							window.close();
							return 0;
						}
						if (event.type == sf::Keyboard::Escape) {
							// free(fz);
							free(pixels);
							window.close();
							return 0;
						}
						if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
							sf::Vector2i mousepos = sf::Mouse::getPosition(window);

							//XMin + (pixels * stepx), XMin + (pixels * stepx)
							//sf::Vector2f plotpos(instance.getMin_x() + (mousepos.x * instance.XstepSize()), instance.getMin_c() + (mousepos.y * instance.YstepSize()));
							instance.zoom(instance.getMin_x() + (mousepos.x * instance.XstepSize()), instance.getMin_c() + (mousepos.y * instance.YstepSize()));

							//instance.zoom(plotpos.x, plotpos.y);
							//camera.setSize(camera.getSize().x / 2, camera.getSize().y / 2);
							instance.setPrecision(instance.getPrecision() + instance.get_zoomIncremental());

							wait = false;


						}
					}

				}




			}


		}

	}







	
	return 0;
}
