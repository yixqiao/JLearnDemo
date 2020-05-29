package me.yixqiao.jlearndemos.main;

import me.yixqiao.jlearn.activations.Linear;
import me.yixqiao.jlearn.activations.ReLU;
import me.yixqiao.jlearn.activations.Sigmoid;
import me.yixqiao.jlearn.activations.Softmax;
import me.yixqiao.jlearn.losses.MeanSquaredError;
import me.yixqiao.jlearn.matrix.Matrix;
import me.yixqiao.jlearn.layers.Dense;
import me.yixqiao.jlearn.layers.InputLayer;
import me.yixqiao.jlearn.losses.CrossEntropy;
import me.yixqiao.jlearn.metrics.Accuracy;
import me.yixqiao.jlearn.metrics.Metric;
import me.yixqiao.jlearn.models.Model;

import java.io.*;
import java.util.ArrayList;

/**
 * Train a network on the MNIST digits dataset.
 */
public class MNIST {
    ArrayList<Matrix> inputsALC = new ArrayList<>();
    ArrayList<Matrix> outputsALC = new ArrayList<>();
    Model model;
    private Matrix inputs;
    private Matrix outputs;
    private Matrix evalInputs;
    private Matrix evalOutputs;

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
                .addLayer(new Dense(64, new ReLU()))
                .addLayer(new Dense(32, new ReLU()))
                .addLayer(new Dense(10, new Sigmoid()));

        model.buildModel(new CrossEntropy());
    }

    private void train() {
        printPredictions();

        ArrayList<Metric> metrics = new ArrayList<>() {{
            add(new Accuracy());
        }};
        model.fit(inputs, outputs, evalInputs, evalOutputs, 0.01, 4, 10, 1, metrics);

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
            }
            dos.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            BufferedReader br = new BufferedReader(new FileReader("datasets/mnist/csv/mnist_test.csv"));
            DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream("datasets/mnist/data/test.dat")));
            String line;
            br.readLine(); // Discard first line
            for (int imgCount = 0; imgCount < 10000; imgCount++) {
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
            }
            dos.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Finished writing to file.");

        System.exit(0);
    }

    private void initInputs() {
        ArrayList<Matrix> inputsAL = new ArrayList<>();
        ArrayList<Matrix> outputsAL = new ArrayList<>();

        // Load training
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

        inputsAL = new ArrayList<>();
        outputsAL = new ArrayList<>();

        // Load testing
        try {
            DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream("datasets/mnist/data/test.dat")));
            for (int i = 0; i < 10000; i++) {
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
        evalInputs = new Matrix(inputsAL.size(), inputsAL.get(0).cols);
        evalOutputs = new Matrix(outputsAL.size(), outputsAL.get(0).cols);

        for (int i = 0; i < inputsAL.size(); i++) {
            for (int j = 0; j < inputsAL.get(i).cols; j++) {
                evalInputs.mat[i][j] = inputsAL.get(i).mat[0][j];
            }
            for (int j = 0; j < outputsAL.get(i).cols; j++) {
                evalOutputs.mat[i][j] = outputsAL.get(i).mat[0][j];
            }
        }

        System.out.println("Finished reading from file.");

        for (int i = 0; i < inputsAL.size(); i += 1000) {
            inputsALC.add(inputsAL.get(i));
            outputsALC.add(outputsAL.get(i));
        }
    }
}

