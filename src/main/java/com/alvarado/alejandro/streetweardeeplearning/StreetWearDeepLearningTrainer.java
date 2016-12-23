package com.alvarado.alejandro.streetweardeeplearning;

import jdk.internal.util.xml.impl.Input;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.ShowImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.io.File;
import java.util.Random;

public class StreetWearDeepLearningTrainer {

    private static Logger log = LoggerFactory.getLogger(StreetWearDeepLearningTrainer.class);

    // Allowed image formats
    private static final String [] allowedExtentions = BaseImageLoader.ALLOWED_FORMATS;

    // Used for reproducibility of results
    private static final long seed = 123;
    private static final Random randomNumGen = new Random(seed);

    // Neural Network parameters
    private static int height = 100;      // Image height
    private static int width = 100;       // Image width
    private static int channels = 3;      // RGB color requires three channels
    private static int numExamples = 100;
    private static int outputNum = 2;     // Results we expect (binary 0 or 1)

    private static int iterations = 1;
    private static double learnRate = 0.01;

    // Training parameters
    private static int batchSize = 64;
    private static int numEpochs = 15;

    public static void main(String[] args) throws Exception {

        log.info("############# Loading Data #############");
        // Get parent directory of our classification directories
        File parentDir = new File(System.getProperty("user.dir"),
                "src/main/resources/images/");
        // Split the parent directory into files (subdirectories)
        FileSplit filesInParentDir = new FileSplit(parentDir,
                allowedExtentions, randomNumGen);
        // Use the following to classify the images based on the subdirectory name.
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        // Randomizes order and balances the paths so that we have equal amount of
        // paths for each data label
        BalancedPathFilter pathFilter = new BalancedPathFilter(randomNumGen,
                allowedExtentions, labelMaker);
        // Split images into train and test subsets: .sample(filter, % in train, % in test)
        InputSplit[] filesInParentDirSplit = filesInParentDir.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInParentDirSplit[0];
        InputSplit testData = filesInParentDirSplit[1];

        // Create a record reader to loop through our images
        // height and width are the sizes our images will be resized to
        ImageRecordReader recordReaderTrain = new ImageRecordReader(height, width, channels, labelMaker);
        ImageRecordReader recordReaderTest = new ImageRecordReader(height, width, channels, labelMaker);


        // Initialize the record readers with the train and test data
        recordReaderTrain.initialize(trainData);
        recordReaderTest.initialize(testData);
        // convert the record readers to iterators for training and testing
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, outputNum);
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReaderTest, batchSize, 1, outputNum);

        log.info("############# Building Model ###########");
        // Our initial neural network, not optimized for this project
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(iterations)
                .activation("relu")
                .weightInit(WeightInit.XAVIER)
                .learningRate(learnRate)
                .updater(Updater.NESTEROVS).momentum(0.98)
                .regularization(true).l2(0.01)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(height * width * channels)
                        .nOut(500)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(500)
                        .nOut(100)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation("softmax")
                        .nIn(100)
                        .nOut(outputNum)
                        .build())
                .pretrain(false).backprop(true)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        log.info("############## Training Model ###############");
        for (int i = 0; i < numEpochs; i++) {
            log.info("Epoch: ", i);
            model.fit(dataIter);
        }

        log.info("############# Evaluate Model ##############");
        Evaluation eval = new Evaluation(outputNum);
        while (testIter.hasNext()) {
            DataSet ds = testIter.next();
            INDArray output = model.output(ds.getFeatureMatrix());
            eval.eval(ds.getLabels(), output);
        }

        log.info(eval.stats());
        log.info("############ Finished Evaluation #############");

    }
}
