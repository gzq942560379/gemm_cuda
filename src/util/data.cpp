#include <random>
#include <fstream>
#include <sstream>
#include "util.h"

void initialize_a_b_c(vector<float> &ha,
					  int size_a,
					  vector<float> &hb,
					  int size_b,
					  vector<float> &hc,
					  vector<float> &hc_gold,
					  int size_c)
{
	srand(size_a);
	for (int i = 0; i < size_a; ++i)
	{
		ha[i] = rand() % 17;
		//      ha[i] = i;
	}
	for (int i = 0; i < size_b; ++i)
	{
		hb[i] = rand() % 5;
		//      hb[i] = 1.0;
	}
	for (int i = 0; i < size_c; ++i)
	{
		hc[i] = rand() % 3;
		//      hc[i] = 1.0;
	}
	hc_gold = hc;
}

void initialize_a_b_c_(vector<float> &ha,
					   int size_a,
					   vector<float> &hb,
					   int size_b,
					   vector<float> &hc,
					   vector<float> &hc_gold,
					   int size_c)
{
	srand(size_a);
	for (int i = 0; i < size_a; ++i)
	{
		// ha[i] = i;
		ha[i] = 1.;
	}
	for (int i = 0; i < size_b; ++i)
	{
		hb[i] = 1.0;
	}
	for (int i = 0; i < size_c; ++i)
	{
		hc[i] = 1.0;
	}
	hc_gold = hc;
}

void matrix_read(vector<float> &hm, int size, const char *file_matrix)
{
	std::ifstream file_m;
	std::string line;
	file_m.open(file_matrix);
	int kk = 0;
	if (file_m.is_open())
	{
		while (size > 0)
		{
			getline(file_m, line);
			std::stringstream is(line);
			int ele;
			while (is >> ele)
			{
				hm[kk] = ele / 1.0;
				kk++;
			}
			size--;
		}
	}
	//std::cout << hm[0] << std::endl;
}