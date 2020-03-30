#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include "myutils.hpp"
#include <iomanip>
#include <cmath>

#define WIDTH 1920
#define HEIGHT 1080

using std::cerr;
using std::endl;

using u8 = unsigned char;
sf::View keep_ratio(const sf::Event::SizeEvent& size, const sf::Vector2u& designedsize);

template<typename DataType>
__global__ void updatePixels(DataType *data, float pulse, float t, float wavefreq) {
	const auto W = gridDim.x * blockDim.x;
	const auto H = gridDim.y * blockDim.y;
	const auto idx = blockDim.x * blockIdx.x + threadIdx.x;
	const auto idy = blockDim.y * blockIdx.y + threadIdx.y;

	// Calculate psi(r) = |cos(pulse * t - wavefreq * r)|
	const float x = idx - W / 2.0;
	const float y = idy - H / 2.0;
	const float r = sqrtf(x * x + y * y);
	const float psi = abs(cosf(pulse * t - wavefreq * r));

	const auto index = 4 * (W * idy + idx);
	data[index + 0] = 0;
	data[index + 1] = 255 * (float(threadIdx.x) / blockDim.x * float(threadIdx.y) / blockDim.y);
	data[index + 2] = 255 * psi;
	data[index + 3] = 255;
}

template<typename DataType>
void updatePixelsCPU(DataType *data, float pulse, float t, float wavefreq, int W, int H) {
	for (int idx = 0; idx < W; ++idx) {
		for (int idy = 0; idy < H; ++idy) {
			// Calculate psi(r) = |cos(pulse * t - wavefreq * r)|
			const float x = idx - W / 2.0;
			const float y = idy - H / 2.0;
			const float r = sqrtf(x * x + y * y);
			const float psi = abs(cosf(pulse * t - wavefreq * r));

			const auto index = 4 * (W * idy + idx);
			data[index + 0] = 0;
			data[index + 1] = 0;
			data[index + 2] = 255 * psi;
			data[index + 3] = 255;
		}
	}
}

int main() {
	/// Init SFML
	sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Test CudaSFML");
	window.setFramerateLimit(60);
	bool vsync = true;
	sf::Texture tex;
	tex.create(WIDTH, HEIGHT);
	sf::Sprite sprite(tex);

	const size_t N = WIDTH * HEIGHT * 4;
	// Array of generated pixels on device.
	u8 *devData;
	// Array to copy generated pixels to on host.
	u8 *hostData;
	// Allocate host and device memory
	MUST(cudaMallocHost(&hostData, N * sizeof(float)))
		MUST(cudaMalloc(&devData, N * sizeof(float)))

		dim3 blockSize(32, 18);
	dim3 gridSize(WIDTH / blockSize.x, HEIGHT / blockSize.y);

	float t = 0; // time
	float pulse = 10;
	float wavefreq = 0.04;
	sf::Clock clock;
	float timeAcc = 0;
	float ms = 0;
	bool cpu = false;

	cudaEvent_t start, end;
	MUST(cudaEventCreate(&start))
		MUST(cudaEventCreate(&end))

		int cycles = 0;
	// Main loop
	while (window.isOpen()) {
		// Event loop
		sf::Event evt;
		while (window.pollEvent(evt)) {
			switch (evt.type) {
			case sf::Event::Closed:
				window.close();
				break;
			case sf::Event::Resized:
				window.setView(keep_ratio(evt.size, sf::Vector2u(WIDTH, HEIGHT)));
			case sf::Event::KeyPressed:
				switch (evt.key.code) {
				case sf::Keyboard::Q:
					window.close();
					break;
				case sf::Keyboard::Add:
					wavefreq += wavefreq / 5.0;
					break;
				case sf::Keyboard::Subtract:
					wavefreq -= wavefreq / 5.0;
					break;
				case sf::Keyboard::V:
					vsync = !vsync;
					window.setFramerateLimit(vsync ? 60 : 0);
					break;
				case sf::Keyboard::C:
					cpu = !cpu;
					break;
				default: break;
				}
			default: break;
			}
		}

		if (!cpu) {
			MUST(cudaEventRecord(start))
				// Generate pixels on device
				updatePixels << <gridSize, blockSize >> > (devData, pulse, t, wavefreq);
			// and copy them to host myutils.hpp
			MUST(cudaMemcpy(hostData, devData, N * sizeof(float), cudaMemcpyDeviceToHost))
				MUST(cudaEventRecord(end))

				MUST(cudaEventSynchronize(end))
				MUST(cudaEventElapsedTime(&ms, start, end))
		}
		else {
			updatePixelsCPU(hostData, pulse, t, wavefreq, WIDTH, HEIGHT);
		}

		tex.update(hostData);

		window.clear();
		window.draw(sprite);
		window.display();

		const auto delta = clock.restart().asSeconds();
		t += delta;
		timeAcc += delta;
		if (!cpu) {
			if (++cycles == 100) {
				std::clog << "[GPU] " << std::setprecision(4) << std::setw(6) << 100.0 / timeAcc << " FPS ("
					<< std::setw(5) << 10 * timeAcc << " ms loop, " << std::setw(5) << ms << " ms CUDA)\n";
				cycles = 0;
				timeAcc = 0;
			}
		}
		else {
			std::clog << "[CPU] " << std::setprecision(4) << std::setw(6) << 1.0 / timeAcc << " FPS ("
				<< std::setw(5) << 1000 * timeAcc << " ms loop)\n";
			cycles = 0;
			timeAcc = 0;
		}
	}

	MUST(cudaFreeHost(hostData))
		MUST(cudaFree(devData))
		MUST(cudaEventDestroy(end))
		MUST(cudaEventDestroy(start))
}

// Handle resizing
sf::View keep_ratio(const sf::Event::SizeEvent& size, const sf::Vector2u& designedsize) {
	sf::FloatRect viewport(0.f, 0.f, 1.f, 1.f);

	const float screenwidth = size.width / static_cast<float>(designedsize.x),
		screenheight = size.height / static_cast<float>(designedsize.y);

	if (screenwidth > screenheight) {
		viewport.width = screenheight / screenwidth;
		viewport.left = (1.f - viewport.width) / 2.f;
	}
	else if (screenwidth < screenheight) {
		viewport.height = screenwidth / screenheight;
		viewport.top = (1.f - viewport.height) / 2.f;
	}

	sf::View view(sf::FloatRect(0, 0, designedsize.x, designedsize.y));
	view.setViewport(viewport);

	return view;
}