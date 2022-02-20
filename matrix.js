class Matrix {
    constructor(rows, cols) {
        this.data = [];
        for (let x = 0; x < rows; x++) {
            this.data.push([]);
            for (let y = 0; y < cols; y++) {
                this.data[x].push([]);
            }
        }

        this.rows = rows;
        this.cols = cols;
    }

    /**
     * Get a specific value in the matrix
     * @param i the row of the value
     *          0 <= i < rows
     * @param j the column of the value
     *          0 <= j < cols
     * @return the value at the specified row and column
     * */
    get(row, col) {
        return this.data[row][col];
    }

    set(row, col, value) {
        this.data[row][col] = value;
    }

    getRows() {
        return this.rows;
    }

    getCols() {
        return this.cols;
    }

    initialize() {
        for (let x = 0; x < this.rows; x++) {
            for (let y = 0; y < this.cols; y++) {
                this.set(x, y, Math.random()*2 - 1);
            }
        }
    }

    initializeZeros() {
        for (let x = 0; x < this.rows; x++) {
            for (let y = 0; y < this.cols; y++) {
                this.set(x, y, 0);
            }
        }
    }

    add(scalar) {
        if (typeof scalar == 'number') {
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    this.data[i][j] += scalar;
                }
            }
        } else if (typeof scalar == 'object' && scalar instanceof Matrix) {
            if (scalar.cols != this.cols || scalar.rows != this.rows) {
                throw "Matrix dimensions do not match";
            }
    
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    this.data[i][j] += scalar.get(i, j);
                }
            }
        }
    }

    multiply(scalar) {
        if (typeof scalar == 'number') {
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    this.data[i][j] *= scalar;
                }
            }
        } else if (typeof scalar == 'object' && scalar instanceof Matrix) {
            for (let i = 0; i < scalar.rows; i++) {
                for (let j = 0; j < scalar.cols; j++) {
                    this.data[i][j] *= scalar.data[i][j];
                }
            }
        } else throw "Multplying with invalid type";
    }

    sigmoid() {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = sigmoid(this.data[i][j]);
            }
        }
    }

    dsigmoid() {
        let temp = new Matrix(this.rows, this.cols);

        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                temp.set(i, j, dsigmoid(this.data[i][j]));
            }
        }

        return temp;
    }

    transpose() {
        return Matrix.transpose(this);
    }

    static subtract(a, b) {
        if (!(a instanceof Matrix && b instanceof Matrix)) return;

        if (a.getCols() != b.getCols() || a.getRows() != b.getRows()) {
            throw "Matrix dimensions do not match";
        }

        let temp = new Matrix(a.getRows(), a.getCols());

        for (let i = 0; i < a.getRows(); i++) {
            for (let j = 0; j < a.getCols(); j++) {
                temp.set(i,j,a.get(i, j) - b.get(i, j));
            }
        }

        return temp;
    }

    static transpose(a) {
        if (!(a instanceof Matrix)) return;

        let temp = new Matrix(a.getCols(), a.getRows());

        for (let i = 0; i < a.rows; i++) {
            for(let j = 0; j < a.cols; j++) {
                temp.set(j, i, a.data[i][j]);
            }
        }

        return temp;
    }

    static multiply(a, b) {
        if (!(a instanceof Matrix && b instanceof Matrix)) throw "Invalid matrix";

        let temp = new Matrix(a.rows,b.cols);

        for (let i = 0; i < temp.rows; i++) {
            for (let j = 0; j < temp.cols; j++) {
                let sum = 0;
                for (let k = 0; k < a.cols; k++) {
                    sum += a.data[i][k] * b.data[k][j];
                }

                temp.data[i][j] = sum;
            }
        }

        return temp;
    }
    
    static map(matrix, func) {
        if (!(matrix instanceof Matrix)) throw "Invalid matrix";

        let temp = new Matrix(matrix.rows, matrix.cols);

        for (let i = 0; i < matrix.rows; i++) {
            for (let j = 0; j < matrix.cols; j++) {
                temp.data[i][j] = func(matrix.data[i][j]);
            }
        }
        
        return temp;
    }

    static fromArray(x, rows=0, cols=0) {
        if (typeof x != 'object') throw "not array";

        if (rows == 0) {
            let temp = new Matrix(x.length,1);
            for (let i = 0; i < x.length; i++) temp.set(i, 0, x[i]);
            return temp;
        } else {
            let temp = new Matrix(rows, cols);
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    temp.set(i, j, x[i * cols + j]);
                }
            }
            return temp;
        }
    }

    toArray() {
        let temp = [];

        for(let i = 0; i < this.rows; i++) {
            for(let j = 0; j < this.cols; j++) {
                temp.push(this.data[i][j]);
            }
        }

        return temp;
    }
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function dsigmoid(x) {
    return x * (1 - x);
}

try {
    module.exports = Matrix;
} catch (e) {
    console.log("Matrix detected no NodeJS, not exporting.");
}