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
import qa.qcri.nadeef.core.utils.CSVTools;
import qa.qcri.nadeef.core.utils.sql.SQLDialectFactory;
import qa.qcri.nadeef.test.NadeefTestBase;
import qa.qcri.nadeef.test.TestDataRepository;

import java.util.List;

/**
 * CleanExecutor test.
 */
@RunWith(value = Parameterized.class)
public class CleanExecutorTest extends NadeefTestBase {
    public CleanExecutorTest(String testConfig_) {
        super(testConfig_);
    }

    @Before
    public void setup() {
        try {
            Bootstrap.start(testConfig);
            NadeefConfiguration.setAlwaysOverride(true);
            CSVTools.dump(
                NadeefConfiguration.getDbConfig(),
                SQLDialectFactory.getNadeefDialectManagerInstance(),
                TestDataRepository.getLocationData1(),
                "LOCATION",
                true
            );
        } catch (Exception ex) {
            ex.printStackTrace();
            Assert.fail(ex.getMessage());
        }
    }

    @After
    public void teardown() {
        Bootstrap.shutdown();
    }

    @Ignore
    static class DetectThread implements Runnable {
        private CleanExecutor cleanExecutor;
        public DetectThread(CleanExecutor cleanExecutor) {
            this.cleanExecutor = cleanExecutor;
        }

        @Override
        public void run() {
            cleanExecutor.detect();
        }
    }

    @Ignore
    static class RepairThread implements Runnable {
        private CleanExecutor cleanExecutor;
        public RepairThread(CleanExecutor cleanExecutor) {
            this.cleanExecutor = cleanExecutor;
        }

        @Override
        public void run() {
            cleanExecutor.repair();
        }
    }

    @Test
    public void progressIndicatorTest() {
        try {
            List<Double> result = Lists.newArrayList();
            CleanPlan cleanPlan = TestDataRepository.getCleanPlan2();
            CleanExecutor executor = new CleanExecutor(cleanPlan);
            Thread thread = new Thread(new DetectThread(executor));
            thread.start();
            while (thread.isAlive()) {
                result.add(executor.getDetectProgress());
                Thread.sleep(30);
            }

            System.out.println("detect");
            for (Double v : result) {
                System.out.println(v);
            }

            Assert.assertEquals(1.0f, executor.getDetectProgress(), 0.0);

            result.clear();
            thread = new Thread(new RepairThread(executor));
            thread.start();
            while (thread.isAlive()) {
                result.add(executor.getRepairProgress());
                Thread.sleep(30);
            }

            System.out.println("repair");
            for (Double v : result) {
                System.out.println(v);
            }
            Assert.assertEquals(1.0f, executor.getRepairProgress(), 0.0);
        } catch (Exception e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }
}
