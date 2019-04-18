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
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import qa.qcri.nadeef.core.datamodel.CleanPlan;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.pipeline.CleanExecutor;
import qa.qcri.nadeef.core.pipeline.NodeCacheManager;
import qa.qcri.nadeef.core.pipeline.UpdateExecutor;
import qa.qcri.nadeef.core.utils.Bootstrap;
import qa.qcri.nadeef.core.utils.CSVTools;
import qa.qcri.nadeef.core.utils.sql.DBInstaller;
import qa.qcri.nadeef.core.utils.sql.SQLDialectFactory;
import qa.qcri.nadeef.test.NadeefTestBase;
import qa.qcri.nadeef.test.TestDataRepository;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.sql.SQLDialect;

import java.util.List;

/**
 * Test the repair phase of NADEEF.
 */
@RunWith(Parameterized.class)
public class RepairPhaseTest extends NadeefTestBase {
    private NodeCacheManager cacheManager;
    private DBConfig sourceConfig;

    public RepairPhaseTest(String config_) {
        super(config_);
    }

    @Before
    public void setup() {
        try {
            Bootstrap.start(testConfig);
            cacheManager = NodeCacheManager.getInstance();
            NadeefConfiguration.setAlwaysOverride(true);

            sourceConfig =
                new DBConfig.Builder()
                    .url("memory:nadeefdb;create=true")
                    .dialect(SQLDialect.DERBYMEMORY)
                    .username("nadeefdb")
                    .build();

            CSVTools.dump(
                sourceConfig,
                SQLDialectFactory.getDialectManagerInstance(sourceConfig.getDialect()),
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
        // Assert.assertEquals(3, cacheManager.getSize());
        Bootstrap.shutdown();
    }

    @Test
    public void test1() {
        try {
            DBInstaller.cleanExecutionDB();
            CleanPlan cleanPlan = TestDataRepository.getCleanPlan3();
            CleanExecutor executor = new CleanExecutor(cleanPlan);
            UpdateExecutor updateExecutor =
                new UpdateExecutor(
                    cleanPlan,
                    NadeefConfiguration.getDbConfig()
                );
            int count = executor.detect().repair().getRepairOutput();
            Assert.assertEquals(1, count);

            executor.shutdown();
            updateExecutor.run();

            count = updateExecutor.getUpdateCellCount();
            Assert.assertEquals(1, count);
        } catch (Exception ex) {
            ex.printStackTrace();
            Assert.fail("exceptions : " + ex.getMessage());
        }
    }

    @Test
    public void test2() {
        try {
            DBInstaller.cleanExecutionDB();
            CleanPlan cleanPlan = TestDataRepository.getCleanPlan();
            CleanExecutor executor = new CleanExecutor(cleanPlan);
            UpdateExecutor updateExecutor =
                new UpdateExecutor(cleanPlan, NadeefConfiguration.getDbConfig());
            int count = executor.detect().repair().getRepairOutput();
            Assert.assertEquals(4, count);

            executor.shutdown();
            updateExecutor.run();

            count = updateExecutor.getUpdateCellCount();
            Assert.assertEquals(2, count);
        } catch (Exception ex) {
            ex.printStackTrace();
            Assert.fail("exceptions : " + ex.getMessage());
        }
    }

    @Test
    public void holisticTest1() {
        try {
            int iterationCount = 0;
            int[] cellChanged = {2, 1, 0};

            List<CleanPlan> cleanPlans = TestDataRepository.getHolisticTestPlan1();
            Assert.assertEquals(2, cleanPlans.size());
            List<CleanExecutor> executors = Lists.newArrayList();
            for (CleanPlan cleanPlan : cleanPlans) {
                executors.add(new CleanExecutor(cleanPlan));
            }

            UpdateExecutor updateExecutor =
                new UpdateExecutor(cleanPlans.get(0), NadeefConfiguration.getDbConfig());
            int changedCell = 0;
            do {
                DBInstaller.cleanExecutionDB();
                for (CleanExecutor executor : executors) {
                    executor.run();
                    executor.getRepairOutput();
                    System.out.println("size: " + cacheManager.getSize());
                }

                updateExecutor.run();
                changedCell = updateExecutor.getUpdateCellCount();
                Assert.assertEquals(cellChanged[iterationCount], changedCell);
                iterationCount ++;
            } while (changedCell != 0);

            for (CleanExecutor executor : executors) {
                executor.shutdown();
            }
            updateExecutor.shutdown();
        } catch (Exception ex) {
            ex.printStackTrace();
            Assert.fail(ex.getMessage());
        }
    }
}
