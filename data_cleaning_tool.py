########################################
# Data Cleaning Tool
# Mohammad Mahdavi
# moh.mahdavi.l@gmail.com
# October 2017
# Big Data Management Group
# TU Berlin
# All Rights Reserved
########################################


########################################
import os
import sys
import json
import re
import string
import subprocess
import random
import pandas
import psycopg2
########################################


########################################
class DataCleaningTool:
    """
    The data cleaning tool class.
    """

    def __init__(self, data_cleaning_tool_dictionary):
        """
        The constructor creates a data cleaning tool.
        """
        self.name = data_cleaning_tool_dictionary["name"]
        self.configuration = data_cleaning_tool_dictionary["configuration"]

    def run(self, d):
        """
        This method takes a dataset to run the data cleaning tool on.
        """
        outputted_cells = {}
        if self.name == "dboost":
            dataset_path = "{}-{}.csv".format(d.name, "".join(
                random.choice(string.ascii_lowercase + string.digits) for _ in range(10)))
            d.write_csv_dataset(dataset_path, d.dataframe)
            self.configuration[0] = "--" + self.configuration[0]
            command = ["./tools/dBoost/dboost/dboost-stdin.py", "-F",
                       ",", "--statistical", "0.5"] + self.configuration + [dataset_path]
            p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            process_output, process_errors = p.communicate()
            tool_results_path = "dboost_output-" + dataset_path
            if os.path.exists(tool_results_path):
                ocdf = pandas.read_csv(tool_results_path, sep=",", header=None, encoding="utf-8", dtype=str,
                                       keep_default_na=False, low_memory=False).apply(lambda x: x.str.strip())
                for i, j in ocdf.get_values().tolist():
                    if int(i) > 0:
                        outputted_cells[(int(i) - 1, int(j))] = ""
                os.remove(tool_results_path)
            os.remove(dataset_path)
        elif self.name == "regex":
            for attribute, pattern, match_type in self.configuration:
                j = d.dataframe.columns.get_loc(attribute)
                for i, value in d.dataframe[attribute].iteritems():
                    if match_type == "OM":
                        if len(re.findall(pattern, value, re.UNICODE)) > 0:
                            outputted_cells[(i, j)] = ""
                    else:
                        if len(re.findall(pattern, value, re.UNICODE)) == 0:
                            outputted_cells[(i, j)] = ""
        elif self.name == "katara":
            dataset_path = "{}-{}.csv".format(d.name, "".join(
                random.choice(string.ascii_lowercase + string.digits) for _ in range(10)))
            d.write_csv_dataset(dataset_path, d.dataframe)
            command = ["java", "-classpath",
                       "$JAVA_HOME/jre/lib/charsets.jar:$JAVA_HOME/jre/lib/ext/cldrdata.jar:"
                       "$JAVA_HOME/jre/lib/ext/dnsns.jar:$JAVA_HOME/jre/lib/ext/icedtea-sound.jar:"
                       "$JAVA_HOME/jre/lib/ext/jaccess.jar:$JAVA_HOME/jre/lib/ext/localedata.jar:"
                       "$JAVA_HOME/jre/lib/ext/nashorn.jar:$JAVA_HOME/jre/lib/ext/sunec.jar:"
                       "$JAVA_HOME/jre/lib/ext/sunjce_provider.jar:$JAVA_HOME/jre/lib/ext/sunpkcs11.jar:"
                       "$JAVA_HOME/jre/lib/ext/zipfs.jar:$JAVA_HOME/jre/lib/jce.jar:$JAVA_HOME/jre/lib/jsse.jar:"
                       "$JAVA_HOME/jre/lib/management-agent.jar:$JAVA_HOME/jre/lib/resources.jar:$JAVA_HOME/jre/lib/rt.jar:"
                       "./tools/KATARA/out/test/test:./tools/KATARA/jar_files/commons-lang3-3.7-test-sources.jar:"
                       "./tools/KATARA/jar_files/commons-lang3-3.7-tests.jar:./tools/KATARA/jar_files/commons-lang3-3.7-sources.jar:"
                       "./tools/KATARA/jar_files/commons-lang3-3.7.jar:./tools/KATARA/jar_files/idea_rt.jar:"
                       "./tools/KATARA/jar_files/SimplifiedKATARA.jar:./tools/KATARA/jar_files/commons-lang3-3.7-javadoc.jar:"
                       "./tools/KATARA/jar_files/super-csv-2.4.0.jar", "simplied.katara.SimplifiedKATARAEntrance"]
            knowledge_base_path = os.path.abspath(self.configuration[0])
            p = subprocess.Popen(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
            p.communicate(dataset_path + "\n" + knowledge_base_path + "\n")
            tool_results_path = "katara_output-" + dataset_path
            if os.path.exists(tool_results_path):
                ocdf = pandas.read_csv(tool_results_path, sep=",", header=None, encoding="utf-8", dtype=str,
                                       keep_default_na=False, low_memory=False).apply(lambda x: x.str.strip())
                for i, j, v in ocdf.get_values().tolist():
                    try:
                        v = v.decode("utf-8")
                    except UnicodeEncodeError:
                        pass
                    outputted_cells[(int(i) - 1, int(j))] = v
                os.remove(tool_results_path)
            if os.path.exists("crowdclient-runtime.log"):
                os.remove("crowdclient-runtime.log")
            os.remove(dataset_path)
        elif self.name == "nadeef":
            # ---------- Prepare Dataset and Clean Plan ----------
            dataset_path = "{}_{}.csv".format(d.name, "".join(
                random.choice(string.ascii_lowercase + string.digits) for _ in range(10)))
            column_index = {a: d.dataframe.columns.get_loc(a) for a in d.dataframe.columns}
            temp_dataframe = d.dataframe.copy()
            temp_dataframe.rename(columns={a: a + " varchar(20000)" for a in temp_dataframe.columns}, inplace=True)
            d.write_csv_dataset(dataset_path, temp_dataframe)
            actual_nadeef_parameters = [{"type": "fd", "value": [" | ".join(param)]} for param in self.configuration]
            nadeef_clean_plan = {
                "source": {
                    "type": "csv",
                    "file": [os.path.abspath(dataset_path)]
                },
                "rule": actual_nadeef_parameters
            }
            nadeef_clean_plan_path = dataset_path + "-nadeef_clean_plan.json"
            nadeef_clean_plan_file = open(nadeef_clean_plan_path, "w")
            json.dump(nadeef_clean_plan, nadeef_clean_plan_file)
            nadeef_clean_plan_file.close()
            # ---------- Connect to the Database ----------
            nadeef_configuration_file = open(os.path.join("tools", "NADEEF", "nadeef.conf"), "r")
            nadeef_configuration = nadeef_configuration_file.read()
            postgres_username = re.findall("database.username = ([\w\d]+)", nadeef_configuration, flags=re.IGNORECASE)[0]
            postgres_password = re.findall("database.password = ([\w\d]+)", nadeef_configuration, flags=re.IGNORECASE)[0]
            nadeef_configuration_file.close()
            connection = psycopg2.connect(dbname="nadeef", host="localhost", user=postgres_username, password=postgres_password)
            cursor = connection.cursor()
            # ---------- Start Data Cleaning ----------
            p = subprocess.Popen(["./nadeef.sh"], cwd=os.path.join("tools", "NADEEF"), stdout=subprocess.PIPE,
                                 stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
            process_output, process_errors = p.communicate("load ../../{}\ndetect\nrepair\nexit\n".format(nadeef_clean_plan_path))
            # tool_results_path = re.findall("INFO: Export to (.*csv)", process_output)[0]
            table_name = "TB_" + dataset_path[:-4].upper()
            cursor.execute("""SELECT * from violation WHERE tablename = '{}';""".format(table_name))
            violation_results = cursor.fetchall()
            for row in violation_results:
                i = int(row[3])
                j = column_index[row[4]]
                outputted_cells[(i - 1, j)] = ""
            cursor.execute("""SELECT * from repair WHERE c1_tablename = '{}';""".format(table_name))
            repair_results = cursor.fetchall()
            for row in repair_results:
                i_1 = int(row[2])
                j_1 = column_index[row[4]]
                v_1 = row[5].decode("utf-8")
                i_2 = int(row[7])
                j_2 = column_index[row[9]]
                v_2 = row[10].decode("utf-8")
                # NOTE: Assume the second cell value is the correct one!
                outputted_cells[(i_1 - 1, j_1)] = v_2
                outputted_cells[(i_2 - 1, j_2)] = v_2
            # ---------- Clean up Current results ----------
            cursor.execute("""DROP TABLE IF EXISTS {}, audit;""".format(table_name))
            cursor.execute("""DELETE FROM violation WHERE tablename = '{}';""".format(table_name))
            cursor.execute("""DELETE FROM repair WHERE c1_tablename = '{}';""".format(table_name))
            connection.commit()
            for f in os.listdir(os.path.join("tools", "NADEEF", "out")):
                if os.path.isfile(os.path.join("tools", "NADEEF", "out", f)):
                    os.remove(os.path.join("tools", "NADEEF", "out", f))
            os.remove(nadeef_clean_plan_path)
            os.remove(dataset_path)
        elif self.name == "fd_checker":
            for l_attribute, r_attribute in self.configuration:
                jl = d.dataframe.columns.get_loc(l_attribute)
                jr = d.dataframe.columns.get_loc(r_attribute)
                value_dictionary = {}
                for i, row in d.dataframe.iterrows():
                    if row[l_attribute] not in value_dictionary:
                        value_dictionary[row[l_attribute]] = {}
                    value_dictionary[row[l_attribute]][row[r_attribute]] = 1
                for i, row in d.dataframe.iterrows():
                    if len(value_dictionary[row[l_attribute]]) > 1:
                        outputted_cells[(i, jl)] = ""
                        outputted_cells[(i, jr)] = ""
        else:
            sys.stderr.write("I do not know this error detection tool!\n")
        return outputted_cells
########################################


########################################
if __name__ == "__main__":

    import dataset

    dataset_dictionary = {
        "name": "toy",
        "path": "datasets/dirty.csv"
    }
    d = dataset.Dataset(dataset_dictionary)

    data_cleaning_tool_dictionary = {
        "name": "nadeef",
        "configuration": [["city", "country"]]
    }
    t = DataCleaningTool(data_cleaning_tool_dictionary)
    print t.run(d)
########################################
