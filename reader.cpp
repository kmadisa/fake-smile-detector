#include <iostream>
#include <fstream>
#include <omp.h>

using namespace std;

void printOut(int**);

/*int main()
{
	char xVal[15];
	char yVal[15];

	int xCord[68];
	int yCord[68];

	ifstream in("../../Image Database/test.txt");

	if(!in)
	{
       cout << "Cannot open file.\n";

	   system("pause");

	   return 1;
	}

	int index = 0;

	do
	{
		in >> xVal;
		in >> yVal;

		cout << "(x, y) -> (" << (int) atof(xVal) << ", " << (int) atof(yVal) << ")" << " index = " << index << endl;

		xCord[index] = (int) atof(xVal);
		yCord[index] = (int) atof(yVal);

		index++;
	}while(in != NULL & index < 68);

	cout << yCord[67] << endl;

	in.close();

	/*int arr[9][9];

	for(int x = 0; x < 9; x++)
	{
		for(int y = 0; y < 9; y++)
		{
			arr[x][y] = x * y;
		}
	}

	printOut(arr);

	system("pause");

	return 0;
}*/

void printOut(int arr[][9])
{
	for(int x = 0; x < 9; x++)
	{
		for(int y = 0; y < 9; y++)
		{
			cout << *(arr[x] + y) << " ";
		}
	}
}