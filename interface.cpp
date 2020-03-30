
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <SFML/Graphics.hpp>
#include <complex>
#include "interface.h"
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
		precision = 10;
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




};
int isMandelbrot(double C_real, double C_complex, settings instance) {
	std::complex<double> Fz(0, 0);
	std::complex<double> complexVar(C_real, C_complex);
	for (int i = 0; i <= instance.getPrecision(); i++) {
		Fz = pow(Fz, 2) + complexVar;
		if ((abs(real(Fz)) >= 2) || ((abs(imag(Fz)) >= 2))) {
			return i;
		}
	}
	return -1;
}

int dointerface()
{
	settings instance;

	sf::RenderWindow window(sf::VideoMode(1920, 1080), "Chaos Explorer");

	sf::View camera;


	camera.setSize(sf::Vector2f(instance.getMax().x - instance.getMin().x, instance.getMax().y - instance.getMin().y));
	std::cout << "X: " << camera.getSize().x << "\nY: " << camera.getSize().y;

	camera.setCenter(0, 0);
	sf::RectangleShape pen(instance.getPen());

	pen.setFillColor(sf::Color::Blue);
	while (window.isOpen())
	{
		std::cout << "system 1: \n";

		window.setView(camera);

		sf::Event event;
		sf::RectangleShape background;
		background.setSize(sf::Vector2f(1, 1));
		background.setSize(sf::Vector2f((instance.getMax().x - instance.getMin().x) * (0.15), (instance.getMax().y - instance.getMin().y) * (0.15)));

		background.setFillColor(sf::Color(180, 180, 180, 255));
		background.setPosition(sf::Vector2f(instance.getMax().x - background.getSize().x, instance.getMin().y));
		std::cout << "HUD Position: " << background.getPosition().x << std::endl << background.getPosition().y << std::endl;





		int count = 0;
		for (double C_real = instance.getMin().x; C_real <= instance.getMax().x; C_real = C_real + instance.XstepSize()) {
			count++;

			for (double C_complex = instance.getMin().y; C_complex <= instance.getMax().y; C_complex = C_complex + instance.YstepSize()) {


				//std::cout << "Point: " << C_real << std::endl << C_complex << std::endl;
				int get = isMandelbrot(C_real, C_complex, instance);
				pen.setPosition(C_real, C_complex);

				if (get == -1) {
					pen.setFillColor(sf::Color::Blue);
				}
				else {
					pen.setFillColor(sf::Color(180, 255 * get / instance.getPrecision(), 0, 255));
				}
				window.draw(pen);




				while (window.pollEvent(event))
				{
					if (event.type == sf::Event::Closed)
						window.close();
					if (event.type == sf::Keyboard::Escape)
						window.close();
				}








			}
			window.display();
		}
		//window.display();
		bool wait = true;
		while (wait) {
			while (window.pollEvent(event))
			{
				if (event.type == sf::Event::Closed)
					window.close();
				if (event.type == sf::Keyboard::Escape)
					window.close();
				if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
					sf::Vector2i mousepos = sf::Mouse::getPosition(window);
					sf::Vector2f plotpos = window.mapPixelToCoords(mousepos);


					instance.zoom(plotpos.x, plotpos.y, camera);
					camera.setSize(camera.getSize().x / 2, camera.getSize().y / 2);
					instance.setPrecision(instance.getPrecision() + 5);
					wait = false;
					pen.setSize(instance.getPen());


				}
			}





		}
	}
	return 0;
}