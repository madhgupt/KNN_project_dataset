#include <iostream>
#include <armadillo>
#include <sstream>
#include <fstream>

using namespace std;
using namespace arma;

int main(int argc, char const *argv[]) {

      // define matrix for the dataset -> X and Y
      mat data_Y , data_X;

      // importing the data
      data_Y.load("Y.csv" , csv_ascii);
      data_X.load("X.csv" , csv_ascii);

      // find no of unique labels
      vec no_of_labels = unique(data_Y);

      // array of Cols for storing indices of similar labels
      uvec index_of_labels[no_of_labels.n_rows];

      // finding the indices of unique labels in Y
      for(int i=0 ; i<no_of_labels.n_rows ; i++){
            index_of_labels[i] = find(data_Y == no_of_labels[i]);
      }

      // array of matrices to saperate X into different labels
      mat data_labels[no_of_labels.n_rows];
      for(int i=0 ; i<no_of_labels.n_rows ; i++){
            data_labels[i] = data_X.rows(index_of_labels[i]);
      }

      // array of matrices to store kmean
      mat kmean_matrix[no_of_labels.n_rows];
      for (int i = 0; i < no_of_labels.n_rows; i++) {
            // running Kmean algo on every unique labeled set
            bool status = kmeans(kmean_matrix[i], data_labels[i].t() , 100, random_subset, 100, true);
            if(status == false){
                  cout << "clustering failed" << endl;
            }
            cout << kmean_matrix[i].n_rows << "\t" << kmean_matrix[i].n_cols << endl;
      }

      return 0;
}
//g++ -std=c++11 file -larmadillo
// -fopenmp
