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

package qa.qcri.nadeef.test;

import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.List;

/**
 * Test base class to provide Regression testing facilities.
 */
public abstract class NadeefTestBase {
    protected String testConfig;
    public NadeefTestBase(String testConfig_) {
        this.testConfig = testConfig_;
    }

    @Parameterized.Parameters
    public static List<String[]> configs() {
        String[][] data;
        // regression test check
        if (System.getProperty("regression") != null) {
            data =
                new String[][] {
                    { TestDataRepository.DerbyConfig },
                    { TestDataRepository.PostgresConfig },
                    { TestDataRepository.MySQLConfig}
                };
        } else {
            data =
                new String[][] {
                    { TestDataRepository.PostgresConfig },
                };
        }
        return Arrays.asList(data);
    }
}
