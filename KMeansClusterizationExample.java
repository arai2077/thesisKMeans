/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.ignite.examples.ml.clustering;

import java.util.Arrays;
import java.util.UUID;
import javax.cache.Cache;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.Ignition;
import org.apache.ignite.cache.affinity.rendezvous.RendezvousAffinityFunction;
import org.apache.ignite.cache.query.QueryCursor;
import org.apache.ignite.cache.query.ScanQuery;
import org.apache.ignite.configuration.CacheConfiguration;
import org.apache.ignite.ml.dataset.impl.cache.CacheBasedDatasetBuilder;
import org.apache.ignite.ml.knn.classification.KNNClassificationTrainer;
import org.apache.ignite.ml.math.Tracer;
import org.apache.ignite.ml.math.distances.EuclideanDistance;
import org.apache.ignite.ml.math.distances.DistanceMeasure;
import org.apache.ignite.ml.math.impls.vector.DenseLocalOnHeapVector;
import org.apache.ignite.ml.clustering.kmeans.KMeansModel;
import org.apache.ignite.ml.clustering.kmeans.KMeansTrainer;
import org.apache.ignite.thread.IgniteThread;
import java.util.ArrayList;
import java.util.List;
import java.lang.Double;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.io.IOException;
import java.util.regex.Pattern;
import java.io.FileWriter;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

/**
 * Run kNN multi-class classification trainer over distributed dataset.
 *
 * @see KNNClassificationTrainer
 */
public class KMeansClusterizationExample {
    /** Run example. */
    public static void main(String[] args) throws InterruptedException {
        System.out.println();
        System.out.println(">>> KMeans clustering algorithm over cached dataset usage example started.");

        int K_value = Integer.parseInt(args[0]);
        System.out.println(">>> K value: " + K_value);

        // Start ignite grid.
        try (Ignite ignite = Ignition.start("examples/config/example-ignite.xml")) {
            System.out.println(">>> Ignite grid started.");

            // Start ignite thread
            IgniteThread igniteThread = new IgniteThread(ignite.configuration().getIgniteInstanceName(),
                KMeansClusterizationExample.class.getSimpleName(), () -> {
                IgniteCache<Integer, double[]> dataCache = getTestCache(ignite);

                // KMeans trainer, input seed and number of cluster centroids
                KMeansTrainer trainer = new KMeansTrainer()
                    .withSeed(7867L)
                    .withMaxIterations(100)
                    .withK(K_value);

                KMeansModel mdl = trainer.fit(
                    new CacheBasedDatasetBuilder<>(ignite, dataCache),
                    (k, v) -> Arrays.copyOfRange(v, 1, v.length),
                    (k, v) -> v[0]
                );

                // Print out cluster centroids
                System.out.println(">>> KMeans centroids");
                for (int i = 0; i < K_value; i++) {
                    Tracer.showAscii(mdl.centers()[i]);
                }

                System.out.println(">>>");

                System.out.println(">>> -----------------------------------");
                System.out.println(">>> | Cluster #\t| Coordinates\t\t\t|");
                System.out.println(">>> -----------------------------------");

                int amountOfErrors = 0;
                int totalAmount = 0;

                // Initialize distance variable for calculating SSE
                EuclideanDistance distance = new EuclideanDistance();

                // Map of cluster centroids and the distances to its datapoints
                List<List<Double>> clusterMap = new ArrayList<List<Double>>();
                for (int i = 0; i < K_value; i++) {
                    clusterMap.add(new ArrayList<Double>());
                }

                // Arraylists for output
                List<List<Double>> xAxis = new ArrayList<List<Double>>();
                for (int i = 0; i < K_value; i++) {
                    xAxis.add(new ArrayList<Double>());
                }
                List<List<Double>> yAxis = new ArrayList<List<Double>>();
                for (int i = 0; i < K_value; i++) {
                    yAxis.add(new ArrayList<Double>());
                }
                List<Double> colorScale = new ArrayList<Double>();

                // Output into JSON objects
                // JSONObject xAxisJSON = new JSONObject();
                // JSONObject yAxisJSON = new JSONObject();
                // JSONObject colorScaleJSON = new JSONObject();

                // Calculate predictions and actual labels of dataset entries
                try (QueryCursor<Cache.Entry<Integer, double[]>> observations = dataCache.query(new ScanQuery<>())) {
                    for (Cache.Entry<Integer, double[]> observation : observations) {
                        double[] val = observation.getValue();
                        double[] inputs = Arrays.copyOfRange(val, 1, val.length);
                        double groundTruth = val[0];

                        // Assigning data points to clusters
                        double prediction = mdl.apply(new DenseLocalOnHeapVector(inputs));

                        // Calculate distance between datapoint and its assigned cluster centroid
                        int clusterIndex = new Double(prediction).intValue();
                        double ret = distance.compute(mdl.centers()[clusterIndex],inputs);

                        // Assign distance to the clustermap
                        clusterMap.get(clusterIndex).add(new Double(ret));

                        totalAmount++;
                        if (groundTruth != prediction) {
                            amountOfErrors++;
                        }

                        // Output arrays
                        xAxis.get(clusterIndex).add(new Double(inputs[0]));
                        yAxis.get(clusterIndex).add(new Double(inputs[1]));
                        colorScale.add(new Double(prediction));

                        // System.out.printf(">>> | %.1f\t| ", prediction);
                        // for (int i = 0; i < inputs.length; i++) {
                        //     System.out.printf("%.1f ", inputs[i]);
                        // }
                        // System.out.print("\n");
                    }

                    // Creating json files of the results
                    // xAxisJSON.put("xAxis", xAxis);
                    // yAxisJSON.put("yAxis", yAxis);
                    // colorScaleJSON.put("colorScale", colorScale);

                    try (FileWriter xAxisFile = new FileWriter("C:/Users/ahti/git/kmeansclustering/elements/graph/js/xAxis.js")) {
                        for (int i = 0; i < xAxis.size(); i++) {
                            xAxisFile.write("var xAxis"+i+" = "+xAxis.get(i)+";\n");
                        }
                    } catch(IOException e) {
                        e.printStackTrace();
                    }
                    try (FileWriter yAxisFile = new FileWriter("C:/Users/ahti/git/kmeansclustering/elements/graph/js/yAxis.js")) {
                        for (int i = 0; i < yAxis.size(); i++) {
                            yAxisFile.write("var yAxis"+i+" = "+yAxis.get(i)+";\n");
                        }
                    } catch(IOException e) {
                        e.printStackTrace();
                    }
                    try (FileWriter colorScaleFile = new FileWriter("C:/Users/ahti/git/kmeansclustering/elements/graph/js/colorScale.js")) {
                        colorScaleFile.write("var colorScale = "+colorScale);
                    } catch(IOException e) {
                        e.printStackTrace();
                    }

                    // Calculating SSE
                    double sum = 0;
                    double average;
                    double sseTotal = 0;
                    double[] sse = new double[K_value];

                    for (int i = 0; i < clusterMap.size(); i++) {
                        sum = 0;
                        sse[i] = 0;

                        for (double d : clusterMap.get(i)) sum += d;
                        average = sum / clusterMap.get(i).size();

                        for (int j = 0; j < clusterMap.get(i).size(); j++) {
                            sse[i] += Math.pow(clusterMap.get(i).get(j) - average, 2);
                        }
                        sseTotal += sse[i];
                        System.out.println("Cluster "+i+" SSE: "+sse[i]);
                    }
                    System.out.println("SSE total: "+sseTotal);


                    System.out.println(">>> ---------------------------------");

                    // System.out.println("\n>>> Absolute amount of errors " + amountOfErrors);
                    // System.out.println("\n>>> Accuracy " + (1 - amountOfErrors / (double)totalAmount));
                }
            });

            igniteThread.start();
            igniteThread.join();
        }
    }

    /**
     * Fills cache with data and returns it.
     *
     * @param ignite Ignite instance.
     * @return Filled Ignite Cache.
     */
    private static IgniteCache<Integer, double[]> getTestCache(Ignite ignite) {

        double[][] data = new double[0][0];
        Pattern spacePattern = Pattern.compile(" ");
        try {
            data = Files.lines(Paths.get("C:/Users/ahti/apache-ignite-fabric-2.5.0-bin/examples/src/main/java/org/apache/ignite/examples/ml/clustering/dataMobile.txt"))
                .map(item -> spacePattern.splitAsStream(item).mapToDouble(Double::parseDouble).toArray())
                .toArray(double[][]::new);

            for (int i = 0; i < data.length; i++) {
                data[i][1] = data[i][1]/100.0;
            }
        }
            catch(IOException e) {
            e.printStackTrace();
        }

        CacheConfiguration<Integer, double[]> cacheConfiguration = new CacheConfiguration<>();
        cacheConfiguration.setName("TEST_" + UUID.randomUUID());
        cacheConfiguration.setAffinity(new RendezvousAffinityFunction(false, 10));

        IgniteCache<Integer, double[]> cache = ignite.createCache(cacheConfiguration);

        for (int i = 0; i < data.length; i++)
            cache.put(i, data[i]);

        return cache;
    }

    /** The Iris dataset. */
    // private static final double[][] data = {
    //     {0,35302,1},
    //     {0,23353,1},
    //     {0,18448,1},
    //     {0,18564,1},
    //     {0,18095,1},
    //     {0,25154,1},
    //     {0,30617,1},
    //     {0,23555,1},
    //     {0,23601,1},
    //     {0,24264,1},
    //     {0,25161,2},
    //     {0,25161,3},
    //     {0,25154,4},
    //     {0,35543,4},
    //     {0,25149,5},
    //     {0,25149,6},
    //     {0,25149,4},
    //     {0,23107,7},
    //     {0,23107,8},
    //     {0,23107,9},
    //     {0,23107,6},
    //     {0,25163,4},
    //     {0,23347,10},
    //     {0,23347,11},
    //     {0,23347,12},
    //     {0,23347,13},
    //     {0,24029,10},
    //     {0,24029,11},
    //     {0,24029,12},
    //     {0,24029,13},
    //     {0,25169,10},
    //     {0,25169,11},
    //     {0,25169,12},
    //     {0,25169,13},
    //     {0,35509,10},
    //     {0,35509,11},
    //     {0,35509,12},
    //     {0,35509,13},
    //     {0,25166,14},
    //     {0,25166,15},
    //     {0,25179,16},
    //     {0,25197,17},
    //     {0,25197,18},
    //     {0,25197,19},
    //     {0,18215,20},
    //     {0,18256,20},
    //     {0,23048,21},
    //     {0,1810,21},
    //     {0,23077,21},
    //     {0,18923,21},
    //     {0,18090,21},
    //     {0,18594,22},
    //     {0,18594,23},
    //     {0,18602,22},
    //     {0,18602,24},
    //     {0,18798,22},
    //     {0,18798,24},
    //     {0,18810,25},
    //     {0,18810,26},
    //     {0,18108,27},
    //     {0,18108,28},
    //     {0,18108,29},
    //     {0,18108,30},
    //     {0,18108,31},
    //     {0,18115,32},
    //     {0,18115,33},
    //     {0,18115,34},
    //     {0,18124,30},
    //     {0,18124,35},
    //     {0,18124,36},
    //     {0,18124,37},
    //     {0,30623,38},
    //     {0,18483,39},
    //     {0,18483,40},
    //     {0,18483,41},
    //     {0,18460,42},
    //     {0,18460,43},
    //     {0,18460,44},
    //     {0,30054,45},
    //     {0,24081,46},
    //     {0,24081,47},
    //     {0,18660,48},
    //     {0,18660,49},
    //     {0,18660,50},
    //     {0,30666,6},
    //     {0,30666,51},
    // };
}
