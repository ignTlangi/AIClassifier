package com.stockclassifier.test;

import classifiers.GPClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Test class for the Genetic Programming classifier
 * This class demonstrates how to use the GPClassifier with stock data
 */
public class GPClassifierTest {
    private GPClassifier classifier;
    private Instances testData;

    @Before
    public void setUp() throws Exception {
        classifier = new GPClassifier();
        
        // Load test data
        DataSource source = new DataSource("data/BTC_test.csv");
        testData = source.getDataSet();
        testData.setClassIndex(testData.numAttributes() - 1);
    }

    @Test
    public void testTrain() throws Exception {
        classifier.train("data/BTC_train.csv");
        assertTrue("Classifier should be trained", classifier.isTrained());
    }

    @Test
    public void testClassify() throws Exception {
        classifier.train("data/BTC_train.csv");
        Instance instance = testData.firstInstance();
        double prediction = classifier.classify(instance);
        assertTrue("Prediction should be between 0 and 1", prediction >= 0 && prediction <= 1);
    }

    @Test(expected = IllegalStateException.class)
    public void testClassifyBeforeTraining() throws Exception {
        Instance instance = testData.firstInstance();
        classifier.classify(instance);
    }

    @Test
    public void testGetModelDescription() throws Exception {
        classifier.train("data/BTC_train.csv");
        String description = classifier.getModelDescription();
        assertNotNull("Model description should not be null", description);
        assertFalse("Model description should not be empty", description.isEmpty());
    }
}