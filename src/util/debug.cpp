#include <iostream>

using std::cout;
using std::endl;

void printMatrix(const char* name, float* A, int m, int n, int lda)
{
	cout << "----------" << name << "----------" << endl;
	int max_size = 15;
	for (int i = 0; i < m && i < max_size; i++)
	{
		for (int j = 0; j < n && j < max_size; j++)
		{
			cout << A[i + j * lda] << " ";
		}
		cout << endl;
	}
}