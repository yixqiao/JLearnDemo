package me.yixqiao.jlearndemos.main;

import me.yixqiao.jlearn.activations.ReLU;
import me.yixqiao.jlearn.activations.Softmax;
import me.yixqiao.jlearn.core.Matrix;
import me.yixqiao.jlearn.layers.Dense;
import me.yixqiao.jlearn.layers.InputLayer;
import me.yixqiao.jlearn.losses.CrossEntropy;
import me.yixqiao.jlearn.metrics.Accuracy;
import me.yixqiao.jlearn.models.Model;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class Iris {
    ArrayList<Matrix> inputsAL = new ArrayList<>();
    ArrayList<Matrix> outputsAL = new ArrayList<>();
    Model model;
    private Matrix inputs;
    private Matrix outputs;

    public static void main(String[] args) {
        Iris iris = new Iris();
        iris.initInputs();
        iris.buildModel();
        iris.train();
    }

    private void buildModel() {
        model = new Model();
        model.addLayer(new InputLayer(4))
                .addLayer(new Dense(32, new ReLU()))
                .addLayer(new Dense(8, new ReLU()))
                .addLayer(new Dense(3, new Softmax()));

        model.buildModel(new CrossEntropy());
    }

    private void train() {
        printPredictions();

        model.fit(inputs, outputs, 0.004, 32, 30000, 1000, new Accuracy());

        printPredictions();
    }

    private void printPredictions() {
        for (int i = 0; i < inputsAL.size(); i += 10) {
            Matrix input = inputsAL.get(i);
            for (int j = 0; j < input.cols; j++) {
                System.out.print(input.mat[0][j]);
                if (j != input.cols - 1) System.out.print(",");
            }
            System.out.print(" :\t");
            Matrix output = model.predict(inputsAL.get(i));
            for (int j = 0; j < output.cols; j++) {
                System.out.print(String.format("%.3f", output.mat[0][j]));
                if (j != output.cols - 1) System.out.print("\t");
            }

            System.out.print("\t-\t");
            for (int j = 0; j < outputs.cols; j++) {
                System.out.print(String.format("%.3f", outputs.mat[i][j]));
                if (j != outputs.cols - 1) System.out.print("\t");
            }

            System.out.println();
        }
    }

    private void initInputs() {
        try (BufferedReader br = new BufferedReader(new FileReader("datasets/IRIS.csv"))) {
            String line;
            br.readLine(); // Discard first line
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                Matrix input = new Matrix(new double[][]{{Double.parseDouble(values[0]), Double.parseDouble(values[1]),
                        Double.parseDouble(values[2]), Double.parseDouble(values[3])}});
                Matrix output = new Matrix(new double[][]{{0, 0, 0}});
                switch (values[4]) {
                    case "Iris-setosa":
                        output.mat[0][0] = 1;
                        break;
                    case "Iris-versicolor":
                        output.mat[0][1] = 1;
                        break;
                    case "Iris-virginica":
                        output.mat[0][2] = 1;
                        break;
                }
                inputsAL.add(input);
                outputsAL.add(output);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
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
    }
}
