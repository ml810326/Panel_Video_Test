#include <iostream>
#include "opencv\cv.h"
#include "opencv\highgui.h"
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <cv.h>
#include <string>
#include <fstream>
#include <string>
#include <omp.h>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  

using namespace cv;
using namespace std;

void BGRtoHSI(Mat input, Mat hsi[3], Mat accu_hue){

	accu_hue = Mat::zeros(input.rows,input.cols,CV_16UC1);
	double aux_hue = 0.0;
	
	//2. Prepare the HSI Channels and Define the Sizes and Depths (8-bit)
	hsi[0] = Mat::zeros(input.rows,input.cols,CV_8UC1);
	hsi[1] = Mat::zeros(input.rows,input.cols,CV_8UC1);
	hsi[2] = Mat::zeros(input.rows,input.cols,CV_8UC1);

	//3. Declares Various Variables 
	double R=0.0, B=0.0, G=0.0; //To store R, G, and B values of a specific in-pixel
	double H=0.0, S=0.0, I=0.0;	//To store H, S, and I values of a specific out-pixel
	double aux_RG=0.0, aux_RB=0.0, aux_GB=0.0; //To store R-G, R-B, G-B
	double aux_c1=0.0, aux_c2=0.0; //To store various values temporarily during calc.
	double theta;	//To store theta (in radian) value (for hue calculation)
	
	//4. RGB to HSI Conversion
	for(int j = 0;j < input.rows; j++){
	for(int i = 0;i < input.cols; i++){
		//a. Acquires the Pixel Values and Normalize Them (0-1)
		B=(double)(input.at<Vec3b>(j,i)[0])/255;	
		G=(double)(input.at<Vec3b>(j,i)[1])/255;	
		R=(double)(input.at<Vec3b>(j,i)[2])/255;	

		aux_c1 = 0.0; aux_c2 = 0.0;
		//b. Fills the Intensity Channel and Normalize the Value to 0-255
		aux_c1 = R+G+B;
		I = aux_c1/3;
		hsi[2].ptr<uchar>(j)[i] = (uchar)(I*255);
			
		//c. Fills the Saturation Channel and Normalize the Value to 0-255
		if(I > 0.0){
		aux_c2 = min_ch(R,G,B);
		S = 1-((3/aux_c1)*aux_c2);
		hsi[1].ptr<uchar>(j)[i] = (uchar)(S*255);}

		aux_c1 = 0.0; aux_c2 = 0.0;
		//d. Fills the Hue Channel and Normalize the Value to 0-255
		if(S > 0.0){
			aux_RG = R-G;aux_RB = R-B;aux_GB = G-B;
			aux_c1 = (aux_RG+aux_RB)/2;
			aux_c2 = sqrt(pow(aux_RG,2.0)+(aux_RB*aux_GB));
			theta = acos(aux_c1/aux_c2);
			if(B<=G){
				aux_hue = theta/(2*PI);
				accu_hue.ptr<ushort>(j)[i] = (unsigned short)(aux_hue * 359.0);
				hsi[0].ptr<uchar>(j)[i] = (uchar)(aux_hue * 255.0);}
			else{
				aux_hue = ((2*PI)-theta)/(2*PI);
				accu_hue.ptr<ushort>(j)[i] = (unsigned short)(aux_hue * 359.0);
				hsi[0].ptr<uchar>(j)[i] = (uchar)(aux_hue * 255.0);}
		}
	}}
}

void BGRtoY(Mat input, Mat Y){
	Y = Mat::zeros(input.rows,input.cols,CV_8UC1);
	double R=0.0, B=0.0, G=0.0;
	for(int j = 0;j < input.rows; j++){
		for(int i = 0;i < input.cols; i++){
			B=(double)(input.at<Vec3b>(j,i)[0]);	
			G=(double)(input.at<Vec3b>(j,i)[1]);	
			R=(double)(input.at<Vec3b>(j,i)[2]);

			Y.ptr<uchar>(j)[i] = (uchar)(0.299*R+0.587*G+0.114*B);
		}
	}
}

void pwr_enhancerV1(Mat input, Mat output, double idSSIM){

	Mat input_gray = Mat::zeros(input.rows, input.cols, CV_8UC1);
	cvtColor(input,input_gray,CV_BGR2GRAY);
	double mean = 0.0;
	for (int j = 0; j < input.rows; j++){
		for (int i = 0; i < input.cols; i++){
			mean += (double)input_gray.ptr<uchar>(j)[i]; 
		}}
	mean = mean / (input.rows * input.cols);

	double delta = 0.0;
	delta = ((idSSIM - 1)+sqrt((1 - (idSSIM * idSSIM))) * mean) / idSSIM;

	double R, G, B;
	for (int j = 0; j < input.rows; j++){
		for (int i = 0; i < input.cols; i++){
			B = (double)input.at<Vec3b>(j,i)[0] - delta; 
			if (B < 0.0){B = 0.0;}
			else if (B > 255.0){B = 255.0;} else{}
			output.at<Vec3b>(j,i)[0] = (uchar)(cvRound(B));
			G = (double)input.at<Vec3b>(j,i)[1] - delta; 
			if (G < 0.0){G = 0.0;}
			else if (G > 255.0){G = 255.0;} else{}
			output.at<Vec3b>(j,i)[1] = (uchar)(cvRound(G));
			R = (double)input.at<Vec3b>(j,i)[2] - delta; 
			if (R < 0.0){R = 0.0;}
			else if (R > 255.0){R = 255.0;} else{}
			output.at<Vec3b>(j,i)[2] = (uchar)(cvRound(R));		
		}}
}

void pwr_enhancerV2(Mat input, Mat output, double mu[3], double men[3]){

	double deltaR = 0.0, deltaG = 0.0, deltaB = 0.0;
	deltaR = mu[0] * men[0];
	deltaG = mu[1] * men[1];
	deltaB = mu[2] * men[2];

	double R, G, B;
	for (int j = 0; j < input.rows; j++){
		for (int i = 0; i < input.cols; i++){
			B = (double)input.at<Vec3b>(j,i)[0] - deltaB; 
			if (B < 0.0){B = 0.0;}
			else if (B > 255.0){B = 255.0;} else{}
			output.at<Vec3b>(j,i)[0] = (uchar)(cvRound(B));
			G = (double)input.at<Vec3b>(j,i)[1] - deltaG; 
			if (G < 0.0){G = 0.0;}
			else if (G > 255.0){G = 255.0;} else{}
			output.at<Vec3b>(j,i)[1] = (uchar)(cvRound(G));
			R = (double)input.at<Vec3b>(j,i)[2] - deltaR; 
			if (R < 0.0){R = 0.0;}
			else if (R > 255.0){R = 255.0;} else{}
			output.at<Vec3b>(j,i)[2] = (uchar)(cvRound(R));		
		}}
}


int main()
{

	VideoCapture vc("C:/Users/NTUST-LiMing/Desktop/2016matlabcode/algorithm/t_video/Avatar - Eywa.mp4");
	int fps = vc.get(CV_CAP_PROP_FPS);
	int fH = vc.get(CV_CAP_PROP_FRAME_HEIGHT);
	int fW = vc.get(CV_CAP_PROP_FRAME_WIDTH);
	VideoWriter vo("black.avi", CV_FOURCC('M', 'J', 'P', 'G'), fps, Size(fW, fH));
	//VideoWriter vw("Power_saving_ssim.avi", CV_FOURCC('M', 'J', 'P', 'G'), fps, Size(fW, fH));
	VideoWriter vm("Power_saving_mine1.avi", CV_FOURCC('M', 'J', 'P', 'G'), fps, Size(fW, fH));
	if (vc.isOpened() == false) {
		cout << "Fail to open video" << endl;
		system("pause");
		return 1;
	}
	if (vo.isOpened() == false) {
		cout << "Fail to write video" << endl;
		system("pause");
		return 1;
	}
	/*if (vw.isOpened() == false) {
		cout << "Fail to write video" << endl;
		system("pause");
		return 1;
	}*/
	if (vm.isOpened() == false) {
		cout << "Fail to write video" << endl;
		system("pause");
		return 1;
	}
	namedWindow("show", CV_WINDOW_AUTOSIZE);
	//namedWindow("output", CV_WINDOW_AUTOSIZE);
	namedWindow("output2", CV_WINDOW_AUTOSIZE);
	Mat frame, dist, dist2;
	vc >> frame;
	Mat gray = Mat::zeros(frame.rows,frame.cols, CV_8UC1);
	BGRtoY(frame, gray);

	double mean = 0.0, OR_th = 0.0, sd = 0.0, pop = (double)(frame.rows*frame.cols);
	for (int j = 0; j < frame.rows; j++){
		for (int i = 0; i < frame.cols; i++){
			mean += (double)gray.ptr<uchar>(j)[i];
		}
	}
	mean /= pop;
	for (int j = 0; j < frame.rows; j++){
		for (int i = 0; i < frame.cols; i++){
			OR_th += pow(((double)gray.ptr<uchar>(j)[i]-mean),2);
		}
	}
	OR_th /= pop;
	OR_th = mean + (0.5*sqrt(OR_th));
	double beta = 25.0;
	cout << "th = " << OR_th << endl;
	
	double A = 0.0;
	A = (255.0 + (OR_th-beta))/2;
	cout << "A = " << A << endl;


	double meanR = 0.0, meanG = 0.0, meanB = 0.0;
	for (int j = 0; j < frame.rows; j++){
		for (int i = 0; i < frame.cols; i++){
			meanR += (double)frame.ptr<uchar>(j,i)[2];
			meanG += (double)frame.ptr<uchar>(j,i)[1];
			meanB += (double)frame.ptr<uchar>(j,i)[0];
		}
	}
	meanR /= pop;
	meanG /= pop;
	meanB /= pop;
	cout << meanR << "," << meanG << "," << meanB << endl;

	double men[3];
	men[0] = meanR;
	men[1] = meanR;
	men[2] = meanR;

	double tR = 0.0, tG = 0.0, tB = 0.0;
	tR = 1-(OR_th/255)*(meanR/A);
	tG = 1-(OR_th/255)*(meanG/A);
	tB = 1-(OR_th/255)*(meanB/A);

	double Rmu = 0.0, Gmu = 0.0, Bmu = 0.0; 
	Rmu = (A - meanR)*(1-tR)/(tR*mean);
	Gmu = (A - meanG)*(1-tG)/(tG*mean);
	Bmu = (A - meanB)*(1-tB)/(tB*mean);
	cout << Rmu << "," << Gmu << "," << Bmu << endl;

	double mu[3];
	mu[0] = Rmu;
	mu[1] = Gmu;
	mu[2] = Bmu;

	char key;
	int count = 0;
	while (frame.empty() == false) 
	{
		count = count + 1;
		imshow("show", frame);
		//Acquires input
		Mat input = frame;
		//Mat output = Mat::zeros(input.rows,input.cols,input.type());
		Mat output2 = Mat::zeros(input.rows,input.cols,input.type());
		Mat output3 = Mat::zeros(input.rows,input.cols,input.type());

		//Setting the SSIM
		double ssim = 0.9;

		/*pwr_enhancerV1(input, output, ssim);*/

		pwr_enhancerV2(input, output2, mu, men);

		key = waitKey(33);
		if (key == 'b')
			break;
		int maxframe = fps*60;
		if (count==maxframe)
			break;
		vo << output3;

		resize(output2, dist2, Size(fW, fH));
		imshow("output2", dist2);
		vm << dist2;

		/*resize(output, dist, Size(fW, fH));
		imshow("output", dist);
		vw << dist;*/
		vc >> frame;
		
	}
	return 0;
}