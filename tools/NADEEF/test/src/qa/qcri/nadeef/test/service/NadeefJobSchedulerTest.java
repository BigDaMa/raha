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

package qa.qcri.nadeef.test.service;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import qa.qcri.nadeef.core.datamodel.CleanPlan;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.utils.Bootstrap;
import qa.qcri.nadeef.service.NadeefJobScheduler;
import qa.qcri.nadeef.service.thrift.TJobStatus;
import qa.qcri.nadeef.service.thrift.TJobStatusType;
import qa.qcri.nadeef.test.NadeefTestBase;
import qa.qcri.nadeef.test.TestDataRepository;

import java.net.InetAddress;

/**
 * JobScheduler test.
 */
@RunWith(Parameterized.class)
public class NadeefJobSchedulerTest extends NadeefTestBase{

    public NadeefJobSchedulerTest(String config) {
        super(config);
    }

    @Before
    public void setup() {
        try {
            Bootstrap.start(testConfig);
            NadeefConfiguration.setAlwaysOverride(true);
        } catch (Exception ex) {
            ex.printStackTrace();
            Assert.fail(ex.getMessage());
        }
    }

    @After
    public void teardown() {
        Bootstrap.shutdown();
    }

    @Test
    public void test1() {
        final int taskNum = 10;

        try {
            String hostname = InetAddress.getLocalHost().getHostName();
            CleanPlan cleanPlan = TestDataRepository.getCleanPlan2();

            NadeefJobScheduler scheduler = NadeefJobScheduler.getInstance();
            String[] keys = new String[taskNum];
            TJobStatus[] jobStatuss = new TJobStatus[taskNum];
            for (int i = 0; i < taskNum; i ++) {
                String key = scheduler.submitDetectJob(cleanPlan);
                keys[i] = key;
                Assert.assertTrue(key.startsWith(hostname));
            }

            int nNotAvailable, nRunning, nWaiting;
            Thread.sleep(3000);
            while (true) {
                nNotAvailable = nRunning = nWaiting = 0;
                for (int i = 0; i < taskNum; i ++) {
                    TJobStatus jobStatus = scheduler.getJobStatus(keys[i]);
                    jobStatuss[i] = jobStatus;
                    TJobStatusType type = jobStatus.getStatus();
                    if (type == TJobStatusType.NOTAVAILABLE) {
                        nNotAvailable ++;
                    } else if (type == TJobStatusType.WAITING) {
                        nWaiting ++;
                    } else if (type == TJobStatusType.RUNNING) {
                        nRunning ++;
                    }
                }
                System.out.println(
                    "NotAvailable: " + nNotAvailable + " Running: " + nRunning + " Waiting: " + nWaiting
                );
                Thread.sleep(1000);
                if (nNotAvailable == taskNum) {
                    break;
                }
            }
        } catch (Exception ex) {
            ex.printStackTrace();
            Assert.fail();
        }
    }
}
