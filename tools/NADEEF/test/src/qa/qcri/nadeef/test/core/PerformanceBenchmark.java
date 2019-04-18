/*
 * QCRI, NADEEF LICENSE
 * NADEEF is an extensible, generalized and easy-to-deploy data cleaning platform built at QCRI.
 * NADEEF means "Clean" in Arabic
 *
 * Copyright (c) 2011-2013, Qatar Foundation for Education, Science and Community Development (on
 * behalf of Qatar Computing Research Institute) having its principle place of business in Doha,
 * Qatar with the registered address P.O box 5825 Doha, Qatar (hereinafter referred to as "QCRI")
 *
 * NADEEF has patent pending nevertheless the following is granted.
 * NADEEF is released under the terms of the MIT License, (http://opensource.org/licenses/MIT).
 */

package qa.qcri.nadeef.test.core;

import com.google.common.collect.Lists;
import org.junit.*;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import qa.qcri.nadeef.core.datamodel.CleanPlan;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.pipeline.CleanExecutor;
import qa.qcri.nadeef.core.utils.Bootstrap;
import qa.qcri.nadeef.core.utils.Violations;
import qa.qcri.nadeef.core.utils.sql.DBInstaller;
import qa.qcri.nadeef.test.NadeefTestBase;
import qa.qcri.nadeef.test.TestDataRepository;
import qa.qcri.nadeef.tools.PerfReport;

import java.io.File;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.List;
import java.util.Map;

/**
 * Performance benchmark class.
 */
@RunWith(Parameterized.class)
public class PerformanceBenchmark extends NadeefTestBase {
    private static List<PerfReport.Metric> measuredMetrics;
    private static PrintWriter writer;
    private static final String separator = ",";

    public PerformanceBenchmark(String testConfig) {
        super(testConfig);
    }


    @BeforeClass
    public static void bootstrap() {
        measuredMetrics = Lists.newArrayList();
        measuredMetrics.add(PerfReport.Metric.DetectTime);
        measuredMetrics.add(PerfReport.Metric.DetectTimeOnly);
        measuredMetrics.add(PerfReport.Metric.DetectThreadCount);
        measuredMetrics.add(PerfReport.Metric.Blocks);
        measuredMetrics.add(PerfReport.Metric.DetectCount);
        measuredMetrics.add(PerfReport.Metric.IteratorTime);
        measuredMetrics.add(PerfReport.Metric.DBLoadTime);
        measuredMetrics.add(PerfReport.Metric.DBConnectionCount);
        measuredMetrics.add(PerfReport.Metric.SourceIndexCreationCount);
        measuredMetrics.add(PerfReport.Metric.NadeefDBConnectionCount);
        measuredMetrics.add(PerfReport.Metric.SourceDBConnectionCount);
        measuredMetrics.add(PerfReport.Metric.ViolationExport);
        measuredMetrics.add(PerfReport.Metric.ViolationExportTime);
        SimpleDateFormat dateFormat = new SimpleDateFormat("MMddHHmmss");
        Calendar calendar = Calendar.getInstance();
        String outputFileName = "benchmark" + dateFormat.format(calendar.getTime()) + ".csv";
        File outputFile = new File(outputFileName);
        try {
            outputFile.createNewFile();
            // write the header
            writer = new PrintWriter(outputFile);
            StringBuilder header = new StringBuilder();
            header.append("Name").append(separator);
            header.append("Database");
            for (PerfReport.Metric metric : measuredMetrics) {
                header.append(separator);
                header.append(metric.toString());
            }
            header.append("\n");
            writer.append(header.toString());
        } catch (Exception ex) {
            ex.getMessage();
            Assert.fail(ex.getMessage());
        }
    }

    @AfterClass
    public static void shutdown() {
        writer.flush();
        writer.close();
    }

    @Before
    public void setup() {
        try {
            Bootstrap.start(testConfig);
            NadeefConfiguration.setMaxIterationNumber(1);
            NadeefConfiguration.setAlwaysOverride(true);
            DBInstaller.uninstall(NadeefConfiguration.getDbConfig());
            PerfReport.clear();
        } catch (Exception ex) {
            ex.printStackTrace();
            Assert.fail(ex.getMessage());
        }
    }

    @After
    public void tearDown() {
        Bootstrap.shutdown();
    }

    @Test
    public void hospital80k() {
        CleanExecutor executor = null;
        try {
            CleanPlan plan = TestDataRepository.getPlan("benchmark80k.json").get(0);
            executor = new CleanExecutor(plan);
            executor.detect();
            verifyViolationResult(58912);
            writeToFile("hospital80k");
        } catch (Exception ex) {
            ex.printStackTrace();
            Assert.fail(ex.getMessage());
        } finally {
            if (executor != null)
                executor.shutdown();
        }
    }

    @Test
    public void hospital100k() {
        CleanExecutor executor = null;
        try {
            CleanPlan plan = TestDataRepository.getPlan("benchmark100k.json").get(0);
            executor = new CleanExecutor(plan);
            executor.detect();
            verifyViolationResult(42800);
            writeToFile("hospital100k");
        } catch (Exception ex) {
            ex.printStackTrace();
            Assert.fail(ex.getMessage());
        } finally {
            if (executor != null)
                executor.shutdown();
        }
    }

    @Test
    public void adult30k_1() {
        CleanExecutor executor = null;
        try {
            CleanPlan plan = TestDataRepository.getPlan("benchmark_adult_30k.json").get(0);
            executor = new CleanExecutor(plan);
            executor.detect();
            verifyViolationResult(0);
            writeToFile("Adult30k_1");
        } catch (Exception ex) {
            ex.printStackTrace();
            Assert.fail(ex.getMessage());
        } finally {
            if (executor != null)
                executor.shutdown();
        }
    }

    @Test
    public void adult30k_2() {
        CleanExecutor executor = null;
        try {
            CleanPlan plan = TestDataRepository.getPlan("benchmark_adult_30k_2.json").get(0);
            executor = new CleanExecutor(plan);
            executor.detect();
            verifyViolationResult(33080);
            writeToFile("Adult30k_2");
        } catch (Exception ex) {
            ex.printStackTrace();
            Assert.fail(ex.getMessage());
        } finally {
            if (executor != null)
                executor.shutdown();
        }
    }

    @Test
    public void hospital500k() {
        CleanExecutor executor = null;
        try {
            CleanPlan plan = TestDataRepository.getPlan("benchmark500k.json").get(0);
            executor = new CleanExecutor(plan);
            executor.detect();
            verifyViolationResult(4050052);
            writeToFile("hospital500k");
        } catch (Exception ex) {
            ex.printStackTrace();
            Assert.fail(ex.getMessage());
        } finally {
            if (executor != null)
                executor.shutdown();
        }
    }

    private void writeToFile(String ruleName) {
        Map<PerfReport.Metric, List<Long>> metrics = PerfReport.getMetrics();
        StringBuilder line = new StringBuilder();
        line.append(ruleName).append(separator).append(getDatabaseName());
        // write the data into the output file.
        try {
            for (PerfReport.Metric metric : measuredMetrics) {
                line.append(separator);
                List<Long> values = metrics.get(metric);
                if (values == null) {
                    line.append("NaN");
                } else if (values.size() > 1) {
                    line.append('[');
                    for (Long value : values) {
                        line.append(value).append(' ');
                    }
                    line.append(']');
                } else {
                    line.append(values.get(0));
                }
            }
            line.append("\n");
            writer.append(line.toString());
        } catch (Exception ex) {
            ex.printStackTrace();
            Assert.fail(ex.getMessage());
        }
    }

    private void verifyViolationResult(int expectRow) throws Exception {
        int rowCount =
            Violations.getViolationRowCount(NadeefConfiguration.getDbConfig());
        Assert.assertEquals(expectRow, rowCount);
    }

    private String getDatabaseName() {
        if (testConfig.contains("derby"))
            return "DerbyInMemory";
        if (testConfig.contains("postgres"))
            return "PostgresSQL";
        if (testConfig.contains("mysql"))
            return "MySQL";
        return "Unknown";
    }
}
