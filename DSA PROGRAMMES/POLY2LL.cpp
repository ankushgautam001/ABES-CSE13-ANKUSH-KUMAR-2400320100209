#include <iostream>
using namespace std;

// Node structure for polynomial
struct Node {
    int coeff;   // coefficient
    int pow;     // power
    Node* next;
};

// Function to create a new node
Node* createNode(int coeff, int pow) {
    Node* newNode = new Node();
    newNode->coeff = coeff;
    newNode->pow = pow;
    newNode->next = nullptr;
    return newNode;
}

// Function to insert a term at end
void insertNode(Node*& poly, int coeff, int pow) {
    Node* newNode = createNode(coeff, pow);
    if (poly == nullptr) {
        poly = newNode;
    } else {
        Node* temp = poly;
        while (temp->next != nullptr)
            temp = temp->next;
        temp->next = newNode;
    }
}

// Function to add two polynomials
Node* addPolynomials(Node* poly1, Node* poly2) {
    Node* result = nullptr;
    Node* temp = nullptr;

    while (poly1 != nullptr && poly2 != nullptr) {
        if (poly1->pow == poly2->pow) {
            int sumCoeff = poly1->coeff + poly2->coeff;
            if (sumCoeff != 0) {
                insertNode(result, sumCoeff, poly1->pow);
            }
            poly1 = poly1->next;
            poly2 = poly2->next;
        } else if (poly1->pow > poly2->pow) {
            insertNode(result, poly1->coeff, poly1->pow);
            poly1 = poly1->next;
        } else {
            insertNode(result, poly2->coeff, poly2->pow);
            poly2 = poly2->next;
        }
    }

    // Add remaining terms
    while (poly1 != nullptr) {
        insertNode(result, poly1->coeff, poly1->pow);
        poly1 = poly1->next;
    }
    while (poly2 != nullptr) {
        insertNode(result, poly2->coeff, poly2->pow);
        poly2 = poly2->next;
    }

    return result;
}

// Function to display polynomial
void display(Node* poly) {
    while (poly != nullptr) {
        cout << poly->coeff << "x^" << poly->pow;
        poly = poly->next;
        if (poly != nullptr)
            cout << " + ";
    }
    cout << endl;
}

// Main function
int main() {
    Node* poly1 = nullptr;
    Node* poly2 = nullptr;
    int i,n,coeff,exp;
    // Polynomial 1
    cout<<"Enter number of terms in the polynomial 1\t";
    cin>>n;
    for(i=0;i<n;i++)
    {
          cout<<"Enter the coeff and exp of term "<<i+1<<"\t";
          cin>>coeff>>exp;
          insertNode(poly1, coeff, exp);
    }


    // Polynomial 2
    cout<<"Enter number of terms in the polynomial 2\t";
    cin>>n;
    for(i=0;i<n;i++)
    {
          cout<<"Enter the coeff and exp of term "<<i+1<<"\t";
          cin>>coeff>>exp;
          insertNode(poly2, coeff, exp);
    }
      cout << "First Polynomial: ";
    display(poly1);
    cout << "Second Polynomial: ";
    display(poly2);

    Node* sum = addPolynomials(poly1, poly2);

    cout << "Resultant Polynomial: ";
    display(sum);

    return 0;
};kk