package me.yixqiao.jlearndemos.main;

import me.yixqiao.jlearn.activations.ReLU;
import me.yixqiao.jlearn.activations.Softmax;
import me.yixqiao.jlearn.matrix.Matrix;
import me.yixqiao.jlearn.layers.Dense;
import me.yixqiao.jlearn.layers.InputLayer;
import me.yixqiao.jlearn.losses.CrossEntropy;
import me.yixqiao.jlearn.metrics.Accuracy;
import me.yixqiao.jlearn.models.Model;

import java.io.*;
import java.util.ArrayList;

public class MNIST {
    ArrayList<Matrix> inputsALC = new ArrayList<>();
    ArrayList<Matrix> outputsALC = new ArrayList<>();
    Model model;
    private Matrix inputs;
    private Matrix outputs;

    public static void main(String[] args) {
        MNIST mnist = new MNIST();
        // mnist.writeDataset();
        mnist.initInputs();
        mnist.buildModel();
        mnist.train();
    }

    private void buildModel() {
        model = new Model();
        model.addLayer(new InputLayer(28 * 28))
                .addLayer(new Dense(256, new ReLU()))
                .addLayer(new Dense(128, new ReLU()))
                .addLayer(new Dense(64, new ReLU()))
                .addLayer(new Dense(32, new ReLU()))
                .addLayer(new Dense(10, new Softmax()));

        model.buildModel(new CrossEntropy());
    }

    private void train() {
        printPredictions();

        model.fit(inputs, outputs, 0.005, 64, 100, 1, new Accuracy());

        printPredictions();
    }

    private void printPredictions() {
        for (int i = 0; i < inputsALC.size(); i++) {
            Matrix output = model.predict(inputsALC.get(i));
            for (int j = 0; j < output.cols; j++) {
                System.out.print(String.format("%.3f", output.mat[0][j]));
                if (j != output.cols - 1) System.out.print("\t");
            }

            System.out.print("\t-\t");
            for (int j = 0; j < outputs.cols; j++) {
                System.out.print(String.format("%.3f", outputsALC.get(i).mat[0][j]));
                if (j != outputs.cols - 1) System.out.print("\t");
            }

            System.out.println();
        }
    }

    private void writeDataset() {
        ArrayList<Matrix> inputsAL = new ArrayList<>();
        ArrayList<Matrix> outputsAL = new ArrayList<>();
        // Flattens all images
        try {
            BufferedReader br = new BufferedReader(new FileReader("datasets/mnist/csv/mnist_train.csv"));
            DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream("datasets/mnist/data/train.dat")));
            String line;
            br.readLine(); // Discard first line
            for (int imgCount = 0; imgCount < 60000; imgCount++) {
                line = br.readLine();
                String[] values = line.split(",");
                Matrix output = new Matrix(1, 10);

                output.mat[0][Integer.parseInt(values[0])] = 1;
                dos.writeByte((byte) Integer.parseInt(values[0]));

                Matrix input = new Matrix(1, 28 * 28);
                for (int i = 0; i < 28 * 28; i++) {
                    input.mat[0][i] = Double.parseDouble(values[1 + i]);
                    dos.writeByte((byte) (input.mat[0][i] - 128));
                }

                input.multiplyIP(1.0 / 255);
                inputsAL.add(input);
                outputsAL.add(output);
            }
            dos.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Finished writing to file.");

        inputs = new Matrix(inputsAL.size(), inputsAL.get(0).cols);
        outputs = new Matrix(outputsAL.size(), outputsAL.get(0).cols);

        for (int i = 0; i < inputsAL.size(); i++) {
            for (int j = 0; j < inputsAL.get(i).cols; j++) {
                inputs.mat[i][j] = inputsAL.get(i).mat[0][j];
            }
            for (int j = 0; j < outputsAL.get(i).cols; j++) {
                outputs.mat[i][j] = outputsAL.get(i).mat[0][j];
            }
        }

        for (int i = 0; i < inputsAL.size(); i += 10000) {
            inputsALC.add(inputsAL.get(i));
        }

    }

    private void initInputs() {
        ArrayList<Matrix> inputsAL = new ArrayList<>();
        ArrayList<Matrix> outputsAL = new ArrayList<>();

        try {
            DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream("datasets/mnist/data/train.dat")));
            for (int i = 0; i < 60000; i++) {
                Matrix output = new Matrix(1, 10);
                output.mat[0][dis.readByte()] = 1;

                Matrix input = new Matrix(1, 28 * 28);
                for (int j = 0; j < 28 * 28; j++) {
                    input.mat[0][j] = dis.readByte() + 128;
                }

                input.multiplyIP(1.0 / 255);
                inputsAL.add(input);
                outputsAL.add(output);
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Finished reading from file.");

        inputs = new Matrix(inputsAL.size(), inputsAL.get(0).cols);
        outputs = new Matrix(outputsAL.size(), outputsAL.get(0).cols);

        for (int i = 0; i < inputsAL.size(); i++) {
            for (int j = 0; j < inputsAL.get(i).cols; j++) {
                inputs.mat[i][j] = inputsAL.get(i).mat[0][j];
            }
            for (int j = 0; j < outputsAL.get(i).cols; j++) {
                outputs.mat[i][j] = outputsAL.get(i).mat[0][j];
            }
        }

        for (int i = 0; i < inputsAL.size(); i += 10000) {
            inputsALC.add(inputsAL.get(i));
            outputsALC.add(outputsAL.get(i));
        }
    }
}

