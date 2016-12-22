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
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
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

    public static void main(String[] args) throws Exception {

        log.info("################ Loading Data ################");
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
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

        // Initialize the record reader with the train data
        recordReader.initialize(trainData);
        // convert the record reader to an iterator for training
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 10, 1, outputNum);
        while (dataIter.hasNext()) {
            DataSet ds = dataIter.next();
            System.out.println(ds);
            try {
                Thread.sleep(3000);
            } catch(InterruptedException ex) {
                Thread.currentThread().interrupt();
            }
        }
        recordReader.reset();

    }
}
