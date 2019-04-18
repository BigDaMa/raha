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

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import qa.qcri.nadeef.core.datamodel.CleanPlan;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.pipeline.CleanExecutor;
import qa.qcri.nadeef.core.utils.Bootstrap;
import qa.qcri.nadeef.core.utils.CSVTools;
import qa.qcri.nadeef.core.utils.Violations;
import qa.qcri.nadeef.core.utils.sql.DBInstaller;
import qa.qcri.nadeef.core.utils.sql.SQLDialectFactory;
import qa.qcri.nadeef.test.NadeefTestBase;
import qa.qcri.nadeef.test.TestDataRepository;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.sql.SQLDialect;

import java.util.List;

/**
 * Tests for CleanExecutor.
 */
@RunWith(Parameterized.class)
public class DetectionTest extends NadeefTestBase {
    private CleanExecutor executor;
    public DetectionTest(String testConfig_) {
        super(testConfig_);
    }

    @Before
    public void setup() {
        try {
            Bootstrap.start(testConfig);
            NadeefConfiguration.setMaxIterationNumber(1);
            NadeefConfiguration.setAlwaysOverride(true);
            DBConfig dbConfig = new DBConfig.Builder()
                .url("memory:nadeefdb;create=true")
                .dialect(SQLDialect.DERBYMEMORY)
                .username("nadeefdb")
                .password("nadeefdb")
                .build();
            CSVTools.dump(
                dbConfig,
                SQLDialectFactory.getDialectManagerInstance(SQLDialect.DERBYMEMORY),
                TestDataRepository.getLocationData1(),
                "LOCATION",
                true
            );
            DBInstaller.uninstall(NadeefConfiguration.getDbConfig());
        } catch (Exception ex) {
            ex.printStackTrace();
            Assert.fail(ex.getMessage());
        }
    }

    @After
    public void shutdown() {
        if (executor != null)
            executor.shutdown();
        Bootstrap.shutdown();
    }

    @Test
    public void cleanExecutorTest() {
        try {
            CleanPlan cleanPlan = TestDataRepository.getCleanPlan();
            executor = new CleanExecutor(cleanPlan);
            executor.detect();
            verifyViolationResult(12);
        } catch (Exception e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void cleanExecutorTest2() {
        try {
            CleanPlan cleanPlan = TestDataRepository.getCleanPlan2();
            executor = new CleanExecutor(cleanPlan);
            executor.detect();
            verifyViolationResult(84);
        } catch (Exception e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void cleanExecutorTest3() {
        try {
            CleanPlan cleanPlan = TestDataRepository.getCleanPlan3();
            executor = new CleanExecutor(cleanPlan);
            executor.detect();
            verifyViolationResult(2);
        } catch (Exception e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void cleanExecutorTest4() {
        try {
            CleanPlan cleanPlan = TestDataRepository.getCleanPlan4();
            executor = new CleanExecutor(cleanPlan);
            executor.detect();
            verifyViolationResult(8);
        } catch (Exception e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void cleanExecutorTest5() {
        try {
            List<CleanPlan> cleanPlans = TestDataRepository.getCleanPlan5();
            executor = null;
            for (CleanPlan cleanPlan : cleanPlans) {
                executor = new CleanExecutor(cleanPlan);
                executor.detect();
            }

            if (executor != null) {
                verifyViolationResult(104);
            }
        } catch (Exception e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void cleanExecutorTest6() {
        try {
            CleanPlan cleanPlan = TestDataRepository.getCleanPlan6();
            executor = new CleanExecutor(cleanPlan);
            executor.detect();
            verifyViolationResult(4);
        } catch (Exception e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void cleanExecutorTest7() {
        try {
            CleanPlan cleanPlan = TestDataRepository.getCleanPlan7();
            executor = new CleanExecutor(cleanPlan);
            executor.detect();
            verifyViolationResult(974);
        } catch (Exception e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void cleanExecutorTest8() {
        try {
            CleanPlan cleanPlan = TestDataRepository.getCleanPlan8();
            executor = new CleanExecutor(cleanPlan);
            executor.detect();
            verifyViolationResult(0);
        } catch (Exception e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void cleanExecutorTest9() {
        try {
            CleanPlan cleanPlan = TestDataRepository.getCleanPlan9();
            executor = new CleanExecutor(cleanPlan);
            executor.detect();
            verifyViolationResult(4);
        } catch (Exception e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void cleanExecutorTest10() {
        try {
            CleanPlan cleanPlan = TestDataRepository.getPlan("CleanPlan12.json").get(0);
            executor = new CleanExecutor(cleanPlan);
            executor.detect();
            verifyViolationResult(36);
        } catch (Exception e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }

    private void verifyViolationResult(int expectRow)
        throws Exception {
        int rowCount = Violations.getViolationRowCount(NadeefConfiguration.getDbConfig());
        Assert.assertEquals(expectRow, rowCount);
    }
}
