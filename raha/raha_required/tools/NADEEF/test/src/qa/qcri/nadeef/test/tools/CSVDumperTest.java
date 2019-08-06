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

package qa.qcri.nadeef.test.tools;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.utils.Bootstrap;
import qa.qcri.nadeef.core.utils.CSVTools;
import qa.qcri.nadeef.core.utils.sql.DBConnectionPool;
import qa.qcri.nadeef.core.utils.sql.SQLDialectBase;
import qa.qcri.nadeef.core.utils.sql.SQLDialectFactory;
import qa.qcri.nadeef.test.NadeefTestBase;
import qa.qcri.nadeef.test.TestDataRepository;
import qa.qcri.nadeef.tools.DBConfig;

import java.io.BufferedReader;
import java.io.FileReader;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;

/**
 * CSV Dumper test.
 */
@RunWith(Parameterized.class)
public class CSVDumperTest extends NadeefTestBase {
    private static String tableName;
    private static DBConfig dbConfig;
    private static SQLDialectBase dialectManager;

    public CSVDumperTest(String config) {
        super(config);
    }

    @Before
    public void setUp() {
        try {
            Bootstrap.start(testConfig);
            dbConfig = NadeefConfiguration.getDbConfig();
            dialectManager =
                SQLDialectFactory.getDialectManagerInstance(dbConfig.getDialect());
        } catch (Exception ex) {
            Assert.fail(ex.getMessage());
        }
    }

    @After
    public void tearDown() {
        try (
            Connection conn = DBConnectionPool.createConnection(dbConfig, true);
            Statement stat = conn.createStatement()
        ) {
            stat.execute(dialectManager.dropTable(tableName));
        } catch (Exception ex) {
            Assert.fail(ex.getMessage());
        }

        Bootstrap.shutdown();
    }



    @Test
    public void goodDumpTest() {
        try (Connection conn = DBConnectionPool.createConnection(dbConfig, true);
             Statement stat = conn.createStatement()
        ) {
            tableName =
                CSVTools.dump(dbConfig, dialectManager, TestDataRepository.getDumpTestCSVFile());
            Assert.assertNotNull("tableName cannot be null", tableName);

            BufferedReader reader =
                new BufferedReader(new FileReader(TestDataRepository.getDumpTestCSVFile()));
            int lineCount = 0;
            String line;
            while ((line = reader.readLine()) != null) {
                lineCount ++;
            }
            reader.close();
            ResultSet resultSet = stat.executeQuery("SELECT COUNT(*) FROM " + tableName);
            int rowCount = -1;
            if (resultSet.next()) {
                rowCount = resultSet.getInt(1);
            }

            Assert.assertEquals("Row number is not correct", lineCount - 1, rowCount);
            resultSet.close();
        } catch (Exception ex) {
            ex.printStackTrace();
            Assert.fail(ex.getMessage());
        }
    }
}
