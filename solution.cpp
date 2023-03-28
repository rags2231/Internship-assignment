#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>

using namespace std;

// Define a struct to hold the attributes of each data point
struct DataPoint {
    vector<double> attributes;
    int class_label;
};

// Define a class for the decision tree
class DecisionTree {
public:
    DecisionTree() {}

    void train(string filename) {
        // Read in the data from the CSV file
        ifstream data_file(filename);
        string line;

        while (getline(data_file, line)) {
            // Split the line by commas
            vector<string> tokens;
            string token;
            for (char c : line) {
                if (c == ',') {
                    tokens.push_back(token);
                    token.clear();
                }
                else {
                    token += c;
                }
            }
            tokens.push_back(token);

            // Convert the tokens to doubles
            vector<double> attributes;
            for (int i = 0; i < tokens.size() - 1; i++) {
                attributes.push_back(stod(tokens[i]));
            }

            // Convert the class label to an integer
            int class_label = stoi(tokens.back());

            // Add the data point to the training set
            training_data.push_back({attributes, class_label});
        }

        // Build the decision tree
        root = build_tree(training_data);
    }

    double test() {
        int num_correct = 0;
        int num_total = test_data.size();

        for (DataPoint data_point : test_data) {
            int predicted_label = classify(data_point, root);
            if (predicted_label == data_point.class_label) {
                num_correct++;
            }
        }

        return (double)num_correct / num_total;
    }

    void predict(string filename) {
        // Read in the data from the CSV file
        ifstream data_file(filename);
        ofstream output_file("predictions.csv");
        string line;

        while (getline(data_file, line)) {
            // Split the line by commas
            vector<string> tokens;
            string token;
            for (char c : line) {
                if (c == ',') {
                    tokens.push_back(token);
                    token.clear();
                }
                else {
                    token += c;
                }
            }
            tokens.push_back(token);

            // Convert the tokens to doubles
            vector<double> attributes;
            for (int i = 0; i < tokens.size() - 1; i++) {
                attributes.push_back(stod(tokens[i]));
            }

            // Make a prediction for the data point
            int predicted_label = classify({attributes, -1}, root);

            // Write the prediction to the output file
            output_file << predicted_label << endl;
        }
    }

private:
    vector<DataPoint> training_data;
    vector<DataPoint> test_data;
    struct Node {
        int attribute;
        double threshold;
        int class_label;
        Node* left;
        Node* right;
    };
    Node* root;

    double entropy(vector<DataPoint>& data) {
        // Count the number of examples in each class
        map<int, int> class_counts;
        for (DataPoint data_point : data) {
            class_counts[data_point.class_label]++;
        }

        // Compute the entropy
        double entropy = 0.0;
        for (auto& p : class_counts) {
            double prob = (double)p        .second / data.size();
        entropy -= prob * log2(prob);
    }

    return entropy;
}

double information_gain(vector<DataPoint>& data, int attribute, double threshold) {
    // Split the data into two subsets based on the threshold
    vector<DataPoint> left_subset;
    vector<DataPoint> right_subset;
    for (DataPoint data_point : data) {
        if (data_point.attributes[attribute] <= threshold) {
            left_subset.push_back(data_point);
        }
        else {
            right_subset.push_back(data_point);
        }
    }

    // Compute the information gain
    double entropy_before = entropy(data);
    double entropy_after = (double)left_subset.size() / data.size() * entropy(left_subset)
                          + (double)right_subset.size() / data.size() * entropy(right_subset);
    double information_gain = entropy_before - entropy_after;

    return information_gain;
}

Node* build_tree(vector<DataPoint>& data) {
    // Base case: if all data points belong to the same class, create a leaf node with that class label
    int first_class_label = data.front().class_label;
    bool all_same_class = true;
    for (DataPoint data_point : data) {
        if (data_point.class_label != first_class_label) {
            all_same_class = false;
            break;
        }
    }
    if (all_same_class) {
        return new Node{-1, 0.0, first_class_label, nullptr, nullptr};
    }

    // Find the attribute and threshold that maximize information gain
    int best_attribute = -1;
    double best_threshold = 0.0;
    double best_information_gain = 0.0;
    for (int attribute = 0; attribute < data.front().attributes.size(); attribute++) {
        // Sort the data points by the attribute value
        vector<DataPoint> sorted_data = data;
        sort(sorted_data.begin(), sorted_data.end(), [attribute](DataPoint a, DataPoint b) {
            return a.attributes[attribute] < b.attributes[attribute];
        });

        // Find the threshold that maximizes information gain
        for (int i = 0; i < sorted_data.size() - 1; i++) {
            double threshold = (sorted_data[i].attributes[attribute] + sorted_data[i + 1].attributes[attribute]) / 2.0;
            double information_gain = information_gain(data, attribute, threshold);
            if (information_gain > best_information_gain) {
                best_attribute = attribute;
                best_threshold = threshold;
                best_information_gain = information_gain;
            }
        }
    }

    // Split the data into two subsets based on the best attribute and threshold
    vector<DataPoint> left_subset;
    vector<DataPoint> right_subset;
    for (DataPoint data_point : data) {
        if (data_point.attributes[best_attribute] <= best_threshold) {
            left_subset.push_back(data_point);
        }
        else {
            right_subset.push_back(data_point);
        }
    }

    // Recursively build the left and right subtrees
    Node* left_child = build_tree(left_subset);
    Node* right_child = build_tree(right_subset);

    return new Node{best_attribute, best_threshold, -1, left_child, right_child};
}

int classify(DataPoint data_point, Node* node) {
    // Base case: if the current node is a leaf, return its class label
    if (node->attribute == -1) {
        return node->class_label;
    }

    // Recursively traverse the tree based on the attribute value
    if (data_point.attributes[node->attribute] <= node->threshold) {
        return classify(data_point, node->left_child);
    }
    else {
        return classify(data_point, node->right_child);
    }
}

void print_tree(Node* node, int indent = 0) {
    if (node == nullptr) {
        return;
    }

    for (int i = 0; i < indent; i++) {
        cout << " ";
    }

    if (node->attribute == -1) {
        cout << "Class " << node->class_label << endl;
    }
    else {
        cout << "Attribute " << node->attribute << " <= " << node->threshold << endl;
        print_tree(node->left_child, indent + 2);
        print_tree(node->right_child, indent + 2);
    }
}

double accuracy(vector<DataPoint>& data, Node* root) {
    int num_correct = 0;
    for (DataPoint data_point : data) {
        if (classify(data_point, root) == data_point.class_label) {
            num_correct++;
        }
    }
    return (double)num_correct / data.size();
}

void write_predictions(vector<DataPoint>& data, Node* root, string output_file_path) {
    ofstream output_file(output_file_path);
    output_file << "Prediction" << endl;
    for (DataPoint data_point : data) {
        output_file << classify(data_point, root) << endl;
    }
    output_file.close();
}

int main() {
    // Load the data from the CSV file
    vector<DataPoint> train_data = load_data("TrainingSet.csv");
    vector<DataPoint> test_data = load_data("TestSet (1).csv");

    // Build the decision tree
    Node* root = build_tree(train_data);

    // Print the decision tree
    print_tree(root);

    // Compute the accuracy on the train data
    double train_accuracy = accuracy(train_data, root);
    cout << "Train accuracy: " << train_accuracy << endl;

    // Write the predictions for the test data to a CSV file
    write_predictions(test_data, root, "C:\Users\umar1\Downloads\answer_internship\predictions.csv");

    // Free the memory allocated for the decision tree
    delete_tree(root);

    return 0;
}

