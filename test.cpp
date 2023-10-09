#include <random>
#include <vector>
#include <algorithm>
using namespace std;

const int BOARD_SIZE = 8;
const double alpha = 1;
const double epslion = 0.25;
const int size = 4;

vector<double> dirichlet(const int& size) {
    random_device rd;
    mt19937 gen(rd() * time(NULL));
    vector<double> sample(size);
    double sum = 0;
    for (int i = 0; i < size; i++) {
        gamma_distribution<double> d(alpha, 1);
        sample[i] = d(gen);
        sum += sample[i];
    }
    for (int i = 0; i < sample.size(); i++) {
        sample[i] /= sum;
    }
    return sample;
}

int main()
{
    vector<double> a = dirichlet(size);
    for (int i = 0; i < size; ++i)
        printf("%.4lf ",a[i]);
    putchar('\n');
    return 0;
}