#include<iostream>
#include<cstdlib>
#include<cmath>
#include<fstream>
#include<list>
#include<time.h>
using namespace std;

class Neuron
{
public:
	double* Weights;
	int size; 
	int X;
	int Y;
	int length;
	double nf;
	 Neuron(int x, int y, int length)
	{
		 size = 35;
		 Weights = new double[size];
		 for (int i = 0;i < size; i++)
			 Weights[i] = 0;
		X = x;
		Y = y;
		this->length = length;
		nf = 1000 / log(length);
	}

	 double Gauss(Neuron win, int it)
	{
		 double A = pow((win.X - X), 2);
		 double B = pow(win.Y - Y, 2);
		 double distance =sqrt(A+B);
		return exp(-pow(distance, 2) / (pow(Strength(it), 2)));
	}
	double LearningRate(int it)
	{
		return exp(-it / 1000) * 0.1;
	}
	double Strength(int it)
	{
		return exp(-it / nf) * length;
	}
	double UpdateWeights(double* pattern, Neuron winner, int it)
	{
		double sum = 0;
		for (int i = 0; i < size; i++)
		{
			double d = pattern[i];
			double e = Weights[i];
			double f = d - e;
			double delta = LearningRate(it) * Gauss(winner, it) * (f);
			Weights[i] += delta;
			sum += delta;
		}
		return sum / size;
	}
	Neuron() {
		size = 35;
		Weights = new double[size];
		for (int i = 0;i < size; i++)
			Weights[i] = 0;
		X = 0;
		Y = 0;
		length = 1;
		nf = 1000 / log(length);
	};
};

void setNeuron(Neuron neuron, int x, int y, int length)
{
	neuron.size = 35;
	neuron.Weights = new double[neuron.size];
	for (int i = 0;i < neuron.size; i++)
		neuron.Weights[i] = 0;
	neuron.X = x;
	neuron.Y = y;
	neuron.length = length;
	neuron.nf = 1000 / log(length);
}

class Map
{
public:
	 Neuron** outputs;  
	 int iteration;      
	 int length;        
	 int dimensions;    
	 double** patterns;
	
	 Map(int dimensions, int length)
	{
		 patterns = new double*[20];
		 for (int i = 0; i < 20;i++)
		 {
			 patterns[i] = new double[35];
		 }

		 for (int i = 0; i < 20;i++)
		 {
			 for (int j = 0; j <35; j++)
			 {
				 patterns[i][j] = 0;
			 }
		 }
		this->length = length;
		this->dimensions = dimensions;
		Initialise();
		LoadData();
		NormalisePatterns();
		Train(0.00001);
		DumpCoordinates();
	}
	 void Initialise()
	{
		 outputs = new Neuron*[length];
		 for (int i = 0; i < length; i++)
		 {
			 outputs[i] = new Neuron[length];
		 }

		for (int i = 0; i < length; i++)
		{
			for (int j = 0; j < length; j++)
			{
				setNeuron(outputs[i][j],i, j, length);
				outputs[i][j].Weights = new double[dimensions];
				for (int k = 0; k < dimensions; k++)
				{
					outputs[i][j].Weights[k] = rand()%10+1;
				}
			}
		}
	}
	 void LoadData()
	{
		fstream plik;
		plik.open("zbior_uczacy1.txt");
		if (plik.good())
		{
			int inputSize = 35;
			int patternSize = 20;

			for (int k = 0; k < patternSize; k++)
			{
				for (int i = 0; i < inputSize; i++)
				{
					plik >> patterns[k][i];
				}
			}
		}
		else
		{
			cout << "blad otwarcia pliku!";
		}
	}
	 void NormalisePatterns()
	{
		 for (int j = 0; j < dimensions; j++)
		 {
			 double sum = 0;
			 for (int i = 0; i< 20; i++)
			 { 
				 sum += patterns[i][j];
			 }
			 double average = sum / 20;
			 for (int i = 0; i< 20; i++)
			 {
				 patterns[i][j] = patterns[i][j] / average;
			 }
		 }
	}

	void normalizeData()
	{
		 for (int i = 0; i <20; i++)
		 {
			 double sum = 0;
			 for (int j = 0; j< 35; j++)
			 {
				 sum += pow(patterns[i][j],2);
			 }
			 double a = sqrt(sum);
			 for (int j = 0; j< 20; j++)
			 {
				 patterns[i][j] = patterns[i][j] / a;
			 }
		 }
		 for (int k = 0; k < 20; k++)
		 {
			 for (int i = 0; i < 35; i++)
			 {
				 cout << patterns[k][i] << " ";
			 }
			 cout << endl;
		 }
	 }
	 void Train(double maxError)
	{
		double currentError = 1000000000;
		while (currentError > maxError)
		{
			currentError = 0;
			
			for(int i=0; i<20; i++)
			{
				currentError += TrainPattern(patterns[i]);	
			}
		}
	}
	double TrainPattern(double* pattern)
	{
		double error = 0;
		Neuron winner = Winner(pattern);
		for (int i = 0; i < length; i++)
		{
			for (int j = 0; j < length; j++)
			{
				error += outputs[i][j].UpdateWeights(pattern, winner, iteration);
			}
		}
		iteration++;
		return abs(error / (length * length));
	}
	void DumpCoordinates()
	{
		for (int i=0; i<20; i++)
		{
			Neuron n = Winner(patterns[i]);
			cout << n.X << " " <<  n.Y <<endl;
		}
	}
	Neuron Winner(double* pattern)
	{ 
		Neuron winner;
		int size = 35;
		double min = 5000000;
		for (int i = 0; i < length; i++)
			for (int j = 0; j < length; j++)
			{
				double d = Distance(pattern, outputs[i][j].Weights,size);
				if (d < min)
				{
					min = d;
					winner = outputs[i][j];
				}
			}
		return winner;
	}
	double Distance(double* vector1, double* vector2, int size1)
	{
		double value = 0;
		for (int i = 0; i < size1; i++)
		{
			double D = vector1[i];
			double E = vector2[i];
			double F = D - E;
			value += pow(F, 2);
		}
		return sqrt(value);
	}
};

int main() {
	srand(time(NULL));
	
		Map A(35, 5);
		//Console.ReadLine();
	getchar();
}
