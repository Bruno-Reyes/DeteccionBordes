#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;
#define pi 3.14
#define e 2.72

// Declarando funciones auxiliares.

Mat RGB_escalaGrises(Mat);
vector<vector<float>> crearKernel(int*, int*);
float convolucion(Mat, vector<vector<float>>, int, int, int);
Mat suavizado(Mat, vector<vector<float>>, int);
Mat Ecualizar(Mat);
vector<vector<float>> kernelSobel(vector <int>);
Mat Sobel(Mat);
float convolucionSobel(Mat, vector<vector<float>>, int, int, int);
Mat Orientacion(Mat);
Mat SobelHorizontales(Mat);
Mat SobelVerticales(Mat);

int main() {
	// Leyendo imagen. 
	Mat imgOriginal = imread("C:/Users/bruno/Documents/IA/5to Semestre IIA/Vision Artificial/Imagenes/nebulosa.jpg");
	cout << "Imagen Original: " << imgOriginal.rows << "x" << imgOriginal.cols << endl;
	
	// Convirtiendo imagen de colores a escala de grises
	Mat escalaGrises = RGB_escalaGrises(imgOriginal);
	
	// Se declara sigma y el tamaño del kernel.
	int sigma, tamanoKernel;
	// Se crea el kernel.
	vector<vector<float>> kernel = crearKernel(&tamanoKernel, &sigma);
	// Se suaviza la imagen con el kernel
	Mat suavizada = suavizado(escalaGrises, kernel, tamanoKernel);
	// Se ecualiza la imagen con la funcion ecualizar.
	Mat ecualizada = Ecualizar(suavizada);
	// Aplicando filtro para bordes horizontales
	Mat horizontales = SobelHorizontales(ecualizada);
	// // Aplicando filtro para bordes verticales
	Mat verticales = SobelVerticales(ecualizada);
	// Aplicando filtro sobel
	Mat sobel = Sobel(ecualizada);
	// Obteniendo imagen de orientacion
	//Mat orientacion = Orientacion(ecualizada);

	// Mostrando imagenes.
	namedWindow("Imagen original", WINDOW_AUTOSIZE);
	imshow("Imagen original", imgOriginal);
	
	namedWindow("Imagen Escala de Grises", WINDOW_AUTOSIZE);
	imshow("Imagen Escala de Grises", escalaGrises);

	namedWindow("Imagen Suavizada", WINDOW_AUTOSIZE);
	imshow("Imagen Suavizada", suavizada);

	namedWindow("Imagen Ecualizada", WINDOW_AUTOSIZE);
	imshow("Imagen Ecualizada", ecualizada);

	namedWindow("Bordes Horizontales", WINDOW_AUTOSIZE);
	imshow("Bordes Horizontales", horizontales);

	namedWindow("Bordes Verticales", WINDOW_AUTOSIZE);
	imshow("Bordes Verticales", verticales);

	namedWindow("Bordes Sobel", WINDOW_AUTOSIZE);
	imshow("Bordes Sobel", sobel);

	waitKey(0);
	return 0;
}

// ==============================================================================================================
//                                Convirtiendo imagen RBG a Escala de grises.
// ==============================================================================================================

// Funcion que convierte de RGB a Escala de Grises
Mat RGB_escalaGrises(Mat original) {
	// Se definen las tres variables auxiliares de colores RGB
	double azul, verde, rojo;

	// Se crea una matriz del mismo tamaño que la imagen original donde se almacenara la escala de grises. 
	Mat escalaGrises(original.rows, original.cols, CV_8UC1);

	// Se recorren filas y columnas de la imagen original
	for (int i = 0; i < original.rows; i++){
		for (int j = 0; j < original.cols; j++){
			// En cada pixel, se capturan los valores de cada color de la imagen original
			azul = original.at<Vec3b>(Point(j, i)).val[0];
			verde = original.at<Vec3b>(Point(j, i)).val[1];
			rojo = original.at<Vec3b>(Point(j, i)).val[2];
			// En el mismo pixel pero de la matriz de escala de grises, se calcula el promedio de los tres colores y se agrega el valor. 
			escalaGrises.at<uchar>(Point(j, i)) = uchar((azul + verde + rojo) / 3);

		}
	}
	cout << "Imagen Escala de Grises: " << escalaGrises.rows << "x" << escalaGrises.cols << endl;
	return escalaGrises;
}

// ==============================================================================================================
//Suavizando imagen. 
// ==============================================================================================================

// Creacion del kernel
vector<vector<float>> crearKernel(int* tamanoKernel, int* sigma) {

	// Se requiere la introduccion del valor de sigma y tamaño del kernel
	cout << "Introduzca el valor de sigma (Entero):"; cin >> *sigma;
	cout << "Introduzca el tamaño del kernel (Numero impar):"; cin >> *tamanoKernel;
	//*sigma = 5;
	//*tamanoKernel = 3;

	// Se crea valida que el tamaño del kernel sea impar
	if (*tamanoKernel % 2 == 0) {
		cout << "El valor del tamaño del kernel debe ser impar" << endl;
		exit(1);
	}

	// Se calcula el salto
	int salto = (*tamanoKernel - 1) / 2;
	// Se crea el kernel
	vector<vector<float>> vec(*tamanoKernel, vector<float>(*tamanoKernel, 0));
	cout << "Valor del filtro Gaussiano!" << endl;
	// Se hace el recorrido en el kernel
	for (int i = -salto; i <= salto; i++){
		for (int j = -salto; j <= salto; j++){
			// Se calculan los valores del kernel aplicando la formula
			float resultado = (1 / sqrt((2 * pi * (*sigma) * (*sigma)))) * pow(e, -(pow((i - j),2) / (2 * (*sigma) * (*sigma))));
			vec[i + salto][j + salto] = resultado;
			cout << resultado << "| ";
		}
		cout << endl;
	}

	return vec;
}

// Funcion que aplica el filtro a un pixel con sus vecinos.
float convolucion(Mat original, vector<vector<float>> kernel, int tamanoKernel, int x, int y) {
	// Se obtienen las filas y las columnas de la imagen. 
	int filas = original.rows;
	int columnas = original.cols;

	// Se calcula el salto. 
	int salto = (tamanoKernel - 1) / 2;

	// Se declaran las variables de la suma del filtro, la suma del kernel y se inicializa en 0.
	float totalFiltro = 0;
	float totalKernel = 0;


	// Se hace el recorrido en el kernel.
	for (int i = -salto; i <= salto; i++){
		for (int j = -salto; j <= salto; j++){
			// Obteniendo el valor del kernel .
			float kernelTemporal = kernel[i + salto][j + salto];

			// Creando variables de posicion temporales para evitar del pixeles de fuera de rango.
			int temporalX = x + i;
			int temporalY = y + j;
			float temporal = 0;
			
			// Validando casos de posiciones de pixeles fuera del rango. 
			if (!(temporalX < 0 || temporalX >= columnas || temporalY < 0 || temporalY >= filas)) {
				// Si todo es correcto se obtiene el valor de ese pixel en la posicion X,Y.
				temporal = original.at<uchar>(Point(temporalX, temporalY));
			}

			// Se multiplica valor del kernel y valor del pixel y se agrega a una variable que guarda el total.
			totalFiltro += (kernelTemporal * temporal);
			// Se guarda el total de la suma de los valores del kernel .
			totalKernel += kernelTemporal;
		}
	}
	// Se retorna el valor del cociente que debe asignarse en el pixel.
	return (totalFiltro / totalKernel);
}

// Funcion que aplica el convoluciones a toda la imagen.
Mat suavizado(Mat original, vector<vector<float>> kernel, int tamanoKernel) {
	// Se crea una nueva matriz vacia para almacenar la imagen suavizada. 
	Mat suavizada(original.rows, original.cols, CV_8UC1);
	// Se recorre la imagen pixel por pixel
	for (int i = 0; i < original.rows; i++){
		for (int j = 0; j < original.cols; j++){
			// Se aplica el filtro en cada pixel (con la funcion "convolucion") que devuelve un valor y se asigna a su posicion en la matriz de suavizado.
			suavizada.at<uchar>(Point(j, i)) = uchar(convolucion(original, kernel, tamanoKernel, j, i));
		}
	}
	cout << "Imagen Suavizada: " << suavizada.rows << "x" << suavizada.cols << endl;
	// Retorna la imagen suavizada
	return suavizada;
}

// ==============================================================================================================
//Ecualizando imagen. 
// ==============================================================================================================

Mat Ecualizar(Mat imagen){

	// Se crea el vector que almacenara el histograma
	vector <int> histograma(256,0);

	// Se recorre toda la imagen, sumando 1 en el nivel de gris correspondiente
	for (int i = 0; i < imagen.rows; i++) {
		for (int j = 0; j < imagen.cols; j++) {
			histograma[imagen.at<uchar>(Point(j, i))] += 1;
		}
	}

	// Se calcula la frecuencia relativa de cada nivel de gris para la ecualizacion = pixelesTotalesPornivel / totalPixeles
	int totalPixeles = imagen.rows * imagen.cols;
	vector <float> frecuenciaRelativa(256, 0);
	for (int i = 0; i < histograma.size();i++) {
		frecuenciaRelativa[i] = ((float)(histograma[i])) / ((float)(totalPixeles));
	}

	// Se calculan los nuevos colores de gris =  ((niveles de gris-1)*(sumatoria de frecuencias relativas)) <-- Redondeado 
	vector <int> nuevoGris(256, 0);
	float sumaAcumulada = 0;
	for (int i = 0; i < frecuenciaRelativa.size();i++) {
		sumaAcumulada += frecuenciaRelativa[i];
		nuevoGris[i] = round((255) * (sumaAcumulada));
	}

	// Se crea una nueva matriz que almacene la imagen y simplemente se sustituye el nivel de gris anterior por el nuevo.
	Mat ecualizada(imagen.rows, imagen.cols, CV_8UC1);
	for (int i = 0; i < imagen.rows; i++) {
		for (int j = 0; j < imagen.cols; j++) {
			ecualizada.at<uchar>(Point(j, i)) = uchar(nuevoGris[imagen.at<uchar>(Point(j, i))]);
		}
	}
	cout << "Imagen Ecualizada: " << ecualizada.rows << "x" << ecualizada.cols << endl;
	return ecualizada;
}

// ==============================================================================================================
//Aplicando filtro Sobel
// ==============================================================================================================

// Funcion que recibe vector con valores de los kernel sobel 
vector<vector<float>> kernelSobel(vector <int> vec) {
	// Crea un vector de vectores para emular una matriz 
	vector<vector<float>> kernel(3, vector<float>(3, 0));
	// Contador es una variable auxiliar.
	int contador = 0;
	// Se rellena la matriz con los valores del vector
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			kernel[i][j] = vec[contador];
			contador += 1;
		}
	}
	// Se retorna el kernel
	return kernel;
}

// Funcion que aplica el filtro a un pixel con sus vecinos.
float convolucionSobel(Mat original, vector<vector<float>> kernel, int tamanoKernel, int x, int y) {
	// Se obtienen las filas y las columnas de la imagen. 
	int filas = original.rows;
	int columnas = original.cols;

	// Se calcula el salto. 
	int salto = (tamanoKernel - 1) / 2;

	// Se declaran las variables de la suma del filtro, la suma del kernel y se inicializa en 0.
	float totalFiltro = 0;

	// Se hace el recorrido en el kernel.
	for (int i = -salto; i <= salto; i++) {
		for (int j = -salto; j <= salto; j++) {
			// Obteniendo el valor del kernel .
			float kernelTemporal = kernel[i + salto][j + salto];
			// Creando variables de posicion temporales para evitar del pixeles de fuera de rango.
			int temporalX = x + i;
			int temporalY = y + j;
			float temporal = 0;

			// Validando casos de posiciones de pixeles fuera del rango. 
			if (!(temporalX < 0 || temporalX >= columnas || temporalY < 0 || temporalY >= filas)) {
				// Si todo es correcto se obtiene el valor de ese pixel en la posicion X,Y.
				temporal = original.at<uchar>(Point(temporalX, temporalY));
			}

			// Se multiplica valor del kernel y valor del pixel y se agrega a una variable que guarda el total.
			totalFiltro += (kernelTemporal * temporal);
		}
	}
	if (totalFiltro > 255) {
		totalFiltro = 255;
	}
	if (totalFiltro < 0) {
		totalFiltro = 0;
	}
	// Se retorna el valor del cociente que debe asignarse en el pixel.
	return (totalFiltro);
}

// Aplicando filtro sobel
Mat Sobel(Mat ecualizada) {
	// Se crean una nuevas matrices vacias para almacenar las imagenes de bordes. 
	Mat sobelVertical(ecualizada.rows, ecualizada.cols, CV_8UC1);
	Mat sobelHorizontal(ecualizada.rows, ecualizada.cols, CV_8UC1);
	Mat sobel(ecualizada.rows, ecualizada.cols, CV_8UC1);

	// Se recorre la imagen pixel por pixel para el sobel vertical
	vector <int> vertical{ -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	vector<vector<float>> kernelVertical = kernelSobel(vertical);
	for (int i = 0; i < ecualizada.rows; i++) {
		for (int j = 0; j < ecualizada.cols; j++) {
			// Se aplica el filtro en cada pixel (con la funcion "convolucionSobel") que devuelve un valor y se asigna a su posicion en la matriz de sobel.
			sobelVertical.at<uchar>(Point(j, i)) = uchar(convolucionSobel(ecualizada, kernelVertical, 3, j, i));
		}
	}
	
	// Se recorre la imagen pixel por pixel para el sobel horizontal
	vector <int> horizontal{ -1, -2, -1, 0, 0, 0, 1, 2, 1 };
	vector<vector<float>> kernelHorizontal = kernelSobel(horizontal);
	for (int i = 0; i < ecualizada.rows; i++) {
		for (int j = 0; j < ecualizada.cols; j++) {
			// Se aplica el filtro en cada pixel (con la funcion "convolucionSobel") que devuelve un valor y se asigna a su posicion en la matriz de sobel.
			sobelHorizontal.at<uchar>(Point(j, i)) = uchar(convolucionSobel(ecualizada, kernelHorizontal, 3, j, i));
		}
	}

	// Se aplica |G| con la distancia euclidiana
	for (int i = 0; i < ecualizada.rows; i++) {
		for (int j = 0; j < ecualizada.cols; j++) {
			sobel.at<uchar>(Point(j, i)) = round(sqrt((pow(sobelHorizontal.at<uchar>(Point(j, i)), 2) + pow(sobelVertical.at<uchar>(Point(j, i)), 2))));
		}
	} 
	cout << "Imagen despues de aplicar |G|: " << sobel.rows << "x" << sobel.cols << endl;
	return sobel;
}

// Imagen con bordes verticales
Mat SobelHorizontales(Mat ecualizada) {
	// Se crean una nuevas matrices vacias para almacenar las imagenes de bordes. 
	Mat sobelHorizontal(ecualizada.rows, ecualizada.cols, CV_8UC1);

	// Se recorre la imagen pixel por pixel para el sobel vertical
	vector <int> horizontal{ -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	vector<vector<float>> kernelHorizontal = kernelSobel(horizontal);
	for (int i = 0; i < ecualizada.rows; i++) {
		for (int j = 0; j < ecualizada.cols; j++) {
			// Se aplica el filtro en cada pixel (con la funcion "convolucionSobel") que devuelve un valor y se asigna a su posicion en la matriz de sobel.
			sobelHorizontal.at<uchar>(Point(j, i)) = uchar(convolucionSobel(ecualizada, kernelHorizontal, 3, j, i));
		}
	}

	return sobelHorizontal;
}

// Imagen con bordes horizontales
Mat SobelVerticales(Mat ecualizada) {
	// Se crean una nuevas matrices vacias para almacenar las imagenes de bordes. 
	Mat sobelVertical(ecualizada.rows, ecualizada.cols, CV_8UC1);

	// Se recorre la imagen pixel por pixel para el sobel vertical
	vector <int> vertical{ -1, -2, -1, 0, 0, 0, 1, 2, 1 };
	vector<vector<float>> kernelVertical = kernelSobel(vertical);
	for (int i = 0; i < ecualizada.rows; i++) {
		for (int j = 0; j < ecualizada.cols; j++) {
			// Se aplica el filtro en cada pixel (con la funcion "convolucionSobel") que devuelve un valor y se asigna a su posicion en la matriz de sobel.
			sobelVertical.at<uchar>(Point(j, i)) = uchar(convolucionSobel(ecualizada, kernelVertical, 3, j, i));
		}
	}

	return sobelVertical;
}

// ==============================================================================================================
//Aplicando Canny
// ==============================================================================================================


// Obteniendo la imagen de angulos
Mat Orientacion(Mat ecualizada) {
	// Se crean una nuevas matrices vacias para almacenar las imagenes de bordes. 
	Mat sobelVertical(ecualizada.rows, ecualizada.cols, CV_8UC1);
	Mat sobelHorizontal(ecualizada.rows, ecualizada.cols, CV_8UC1);
	Mat orientacion(ecualizada.rows, ecualizada.cols, CV_8UC1);

	// Se recorre la imagen pixel por pixel para el sobel vertical
	vector <int> vertical{ -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	vector<vector<float>> kernelVertical = kernelSobel(vertical);
	for (int i = 0; i < ecualizada.rows; i++) {
		for (int j = 0; j < ecualizada.cols; j++) {
			// Se aplica el filtro en cada pixel (con la funcion "convolucionSobel") que devuelve un valor y se asigna a su posicion en la matriz de sobel.
			sobelVertical.at<uchar>(Point(j, i)) = uchar(convolucionSobel(ecualizada, kernelVertical, 3, j, i));
		}
	}

	// Se recorre la imagen pixel por pixel para el sobel horizontal
	vector <int> horizontal{ -1, -2, -1, 0, 0, 0, 1, 2, 1 };
	vector<vector<float>> kernelHorizontal = kernelSobel(horizontal);
	for (int i = 0; i < ecualizada.rows; i++) {
		for (int j = 0; j < ecualizada.cols; j++) {
			// Se aplica el filtro en cada pixel (con la funcion "convolucionSobel") que devuelve un valor y se asigna a su posicion en la matriz de sobel.
			sobelHorizontal.at<uchar>(Point(j, i)) = uchar(convolucionSobel(ecualizada, kernelHorizontal, 3, j, i));
		}
	}
	
	// Se obtiene el valor del angulo del gradiente con el arco tangente de Y / X
	for (int i = 0; i < ecualizada.rows; i++) {
		for (int j = 0; j < ecualizada.cols; j++) {
			if ( (float)(sobelHorizontal.at<uchar>(Point(j, i))) == 0.0 ) {
				orientacion.at<uchar>(Point(j, i)) = 0;
			}
			else{
				orientacion.at<uchar>(Point(j, i)) = round((atan((sobelVertical.at<uchar>(Point(j, i)) / sobelHorizontal.at<uchar>(Point(j, i)))) * 360) / (2 * pi));
			}
		}
	}
	return orientacion;
}