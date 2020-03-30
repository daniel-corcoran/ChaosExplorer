//---CUDA BUILD---//
//DANIEL CORCORAN IS A PROGRAMMING GOD


#include <iostream>
#include <SFML/Graphics.hpp>
#include <complex>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuComplex.h>
#include <chrono>
class settings {
	double xMax, xMin, cMax, cMin;	//S is -2
	int precision;
	double resolution_multiplier; //Reccomended is 20
	double penSize;
public:
	settings() {
		xMax = 2;
		xMin = -2;
		cMax = 2;
		cMin = -2;
		precision = 35;
		resolution_multiplier = 1;
	}
	sf::Vector2f getMin() {
		sf::Vector2f min(xMin, cMin);
		return min;
	}
	sf::Vector2f getMax() {
		sf::Vector2f max(xMax, cMax);
		return max;
	}
	sf::Vector2f getPen() {
		sf::Vector2f mypen(((xMax - xMin) / 1920) * resolution_multiplier * 2, ((cMax - cMin) / 1080) * resolution_multiplier);
		return mypen;

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
	void zoom(double x, double y, sf::View &myView) {
		myView.setCenter(sf::Vector2f(x, y));
		myView.setSize(myView.getSize().x / 4, myView.getSize().y / 4);
		xMin = myView.getCenter().x - (myView.getSize().x / 4);
		xMax = myView.getCenter().x + (myView.getSize().x / 4);
		cMin = myView.getCenter().y - (myView.getSize().y / 4);
		cMax = myView.getCenter().y + (myView.getSize().y / 4);
	}
	void setPrecision(int newPrecision) {
		precision = newPrecision;
	}
	int getRes() { return resolution_multiplier; }



};

__global__
void process(int n, int iterations, double Minx, double Miny, double *fz, double x_step, double y_step, int displayX, int displayY) {
	//	std::cout << blockIdx.x << std::endl;

	int id = blockIdx.x*blockDim.x + threadIdx.x;
	fz[id] = 0;
	if (id < n) {
		int i;

		int x_index = id % displayX;
		int y_index = ((id - x_index) / displayX);
		//convert x_index to coord_pos
		double c_r = Minx + (x_index * x_step);
		double c_i = Miny + (y_index * y_step);


		double zr = 0; double zi = 0; double zrsqr = 0; double zisqr = 0;

		for (i = 0; i < iterations; i++) {
			zi = zr * zi;
			zi += zi;
			zi += c_i;
			zr = zrsqr - zisqr + c_r;
			zrsqr = zr * zr;
			zisqr = zi * zi;

			if ((abs(zrsqr) >= 2) || (abs(zisqr) >= 2)) {
				break;
			}

			if (zrsqr + zisqr > 6.0) break;
		}
		fz[id] = i;


	}

}





int main()
{
	settings instance;

	sf::RenderWindow window(sf::VideoMode(1920, 1080), "Chaos Explorer");

	sf::View camera;


	camera.setSize(sf::Vector2f(instance.getMax().x - instance.getMin().x, instance.getMax().y - instance.getMin().y));
	std::cout << "X: " << camera.getSize().x << "\nY: " << camera.getSize().y;

	camera.setCenter(0, 0);
	sf::RectangleShape pen(instance.getPen());

	pen.setFillColor(sf::Color::Blue);


	//Create a giant array for the GPU to do the iterations on

	//Declare giant tables;
	int displayX = ceil(1920 / instance.getRes());
	int displayY = ceil(1080 / instance.getRes());

	int total_size = displayX * displayY; //How many elements in this array?
	std::cout << "arrays purged succesfully";
	double *fz;
	cudaMallocManaged(&fz, total_size * sizeof(double));

	while (window.isOpen()) {







		std::cout << "total length: " << total_size << std::endl;





		for (int i = 0; i < total_size; i++) {
			fz[i] = 0.0f;
		}

		{
			std::cout << "Y points: " << displayY << "\nX points: " << displayX << std::endl;
			//cudaMemcpy(d_fz, fz, total_size * sizeof(double), cudaMemcpyHostToDevice);

			double Minx = instance.getMin().x;
			double Miny = instance.getMin().y;
			int iterations = instance.getPrecision();
			double x_step = instance.XstepSize();
			double y_step = instance.YstepSize();
			auto start = std::chrono::high_resolution_clock::now();
			process << <total_size, 1 >> > (total_size, iterations, Minx, Miny, fz, x_step, y_step, displayX, displayY);
			cudaDeviceSynchronize();
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
			std::cout << "duration: " << duration.count() << std::endl;
			//cudaMemcpy(d_fz, fz, total_size * sizeof(double), cudaMemcpyDeviceToHost);




			std::cout << "system 1: \n";

			window.setView(camera);

			sf::Event event;


			int get;
			int index_x = 0;
			int index_y = 0;
			window.clear(sf::Color::Black);
			for (double C_complex = instance.getMin().y; C_complex < instance.getMax().y; C_complex = C_complex + instance.YstepSize()) {
				index_x = 0;
				for (double C_real = instance.getMin().x; C_real < instance.getMax().x; C_real = C_real + instance.XstepSize()) {

					//std::cout << "-------------\n";
				//	std::cout << "attempted to access index : " << index_x + (displayX * index_y) << std::endl;
					//size_t S = sizeof(fz) / sizeof(fz[0]);
					//std::cout << "out of " << total_size << std::endl;
					//std::cout << "index x " << index_x << "\nindex y " << index_y << std::endl;
					//std::cout << "value: ";


					//std::cout << "out of: " << sizeof(fz) / sizeof(fz[0]) <<std::endl;
					get = fz[index_x + (displayX * index_y)];
					//std::cout << get << std::endl;
					pen.setPosition(C_real, C_complex);
					if (get == instance.getPrecision()) {
						pen.setFillColor(sf::Color::Black);
						//	std::cout << "diverge!" << std::endl;
					}
					else {
						pen.setFillColor(sf::Color(255 * static_cast<double>(get) / instance.getPrecision(), 230 * static_cast<double>(get) / instance.getPrecision(), 255, 255));
					}
					window.draw(pen);
					//window.display();



					while (window.pollEvent(event))
					{
						if (event.type == sf::Event::Closed)
							window.close();
						if (event.type == sf::Keyboard::Escape)
							window.close();
					}







					if (index_x + (displayX * index_y) < total_size) {
						index_x++;
					}

				}
				if (index_x + (displayX * index_y) < total_size - displayX) {
					index_y++;
				}

			}
			window.display();

			//cudaFree(d_fz);
			//window.display();
			std::cout << "render complete" << std::endl;
			bool wait = true;
			while (wait) {
				while (window.pollEvent(event))
				{
					if (event.type == sf::Event::Closed) {
						free(fz);
						window.close();
					}
					if (event.type == sf::Keyboard::Escape) {
						free(fz);
						window.close();
					}
					if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
						sf::Vector2i mousepos = sf::Mouse::getPosition(window);
						sf::Vector2f plotpos = window.mapPixelToCoords(mousepos);


						instance.zoom(plotpos.x, plotpos.y, camera);
						camera.setSize(camera.getSize().x / 2, camera.getSize().y / 2);
						instance.setPrecision(instance.getPrecision() + 20);
						wait = false;
						pen.setSize(instance.getPen());


					}
				}






			}


		}


	}
	return 0;
}