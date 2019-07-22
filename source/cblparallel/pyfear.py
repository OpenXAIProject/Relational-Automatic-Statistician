'''
A set of utilities to talk to the
fear computing cluster and perform
common tasks

@authors: James Robert Lloyd (jrl44@cam.ac.uk)
'''

import pysftp # Wraps up various paramiko calls
from config import * # Various constants such as USERNAME
from util import timeoutCommand
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET

class fear(object):
    '''
    Manages communications with the fear computing cluster
    TODO - efficiently communicate over gate.eng.cam.ac.uk
    '''

    def __init__(self, via_gate=False):
        '''
        Constructor - connects to fear
        '''
        self.via_gate = via_gate
        self.connect()
        
    def __enter__(self):
        return self

    def __exit__(self, _type, value, traceback):
        self.disconnect()
        
    def connect(self):
        '''
        Connect to fear and store connection object
        '''
        if not self.via_gate:
            self._connection = pysftp.Connection('fear', private_key=LOCAL_TO_REMOTE_KEY_FILE)
        else:
            #### FIXME - Currently assumes that necessary port forwarding already in place
            self._connection = pysftp.Connection('localhost', port=HOME_TO_REMOTE_PORT, username=USERNAME, private_key=HOME_TO_REMOTE_KEY_FILE)
            #self._connection = pysftp.Connection('gate.eng.cam.ac.uk', username=USERNAME, password=HOME_TO_GATE_PWD)
        
    def disconnect(self):
        self._connection.close()

    def command(self, cmd):
        #### TODO - Allow port forwarding / tunneling - can this be authenticated / os independent?
        #if self.via_gate:
        #    #### FIXME - This is an inefficient hack
        #    ####       - If anyone knows how to set up the appropriate SSH client object across gate then please implement!!!
        #    cmd = 'ssh -i %(rsa_key)s %(username)s@fear "%(cmd)s"' % {'rsa_key' : GATE_TO_REMOTE_KEY_FILE,
        #                                                              'username' : USERNAME,
        #                                                              'cmd' : cmd}
        success = False
        fail_count = 0
        while not success:
            try:
                output =  self._connection.execute(cmd)
                success = True
            except:
                fail_count += 1
                if fail_count >= 10:
                    raise RuntimeError('Failed to send command 10 times')
                print 'Failed to excute command: %s' % cmd
                print 'Sleeping for a minute'
                time.sleep(60)
        return output
        
    def multiple_commands(self, cmds):
        '''
        Just places semicolons between commands - trying to eak out some extra speed
        '''
        return self.command(' ; '.join(cmds))
        
    def _put(self, localpath, remotepath):
        return self._connection.put(localpath=localpath, remotepath=remotepath)
    
    def _get(self, remotepath, localpath):
        #### FIXME - This occassionally hangs
        self._connection.get(remotepath=remotepath, localpath=localpath)
        
    def copy_to(self, localpath, remotepath, timeout=10, verbose=False):
        self._put(localpath=localpath, remotepath=remotepath)
        #### TODO - Make this operating system independent
        #### TODO - Allow port forwarding / tunneling via gate.eng.cam.ac.uk - can we make authentication work?
        #if not self.via_gate:
        #    #### This scp command is very inefficient
        #    #cmd = 'scp -i %(rsa_key)s %(localpath)s %(username)s@fear:%(remotepath)s' % {'rsa_key' : LOCAL_TO_REMOTE_KEY_FILE,
        #    #                                                                             'localpath' : localpath,
        #    #                                                                             'username' : USERNAME,
        #    #                                                                             'remotepath' : remotepath} 
        #    #return timeoutCommand(cmd=cmd, verbose=verbose).run(timeout=timeout)
        #    self._put(localpath=localpath, remotepath=remotepath)
        #else:
        #    #### FIXME - This is an inefficient hack
        #    ####       - If anyone knows how to set up the appropriate SFTP client object across gate then please implement!!!
        #    
        #    # Put the file on gate
        #    self._put(localpath=localpath, remotepath=os.path.split(localpath)[-1])
        #    # Copy across to fear
        #    cmd = 'ssh -i %(rsa_key)s %(username)s@fear "cat > %(remotepath)s" < %(temppath)s' % {'rsa_key' : GATE_TO_REMOTE_KEY_FILE,
        #                                                                                          'username' : USERNAME,
        #                                                                                          'remotepath' : remotepath,
        #                                                                                          'temppath' : os.path.split(localpath)[-1]}
        #    self._connection.execute(cmd)                                                                                   
        #    # Clear the temporary file
        #    self._connection.execute('rm %s' % os.path.split(localpath)[-1])
        #    
        #    #### This is an even more hacky way to do things
        #    #with open(localpath, 'rb') as local_file:
        #    #    file_contents = local_file.read()
        #    #cmd = 'echo "%s" > %s' % (file_contents, remotepath)
        #    #self.command(cmd) 
    
    def copy_to_temp(self, localpath, timeout=10, verbose=False):
        #### TODO - make this OS independent
        self.copy_to(localpath=localpath, remotepath=os.path.join(REMOTE_TEMP_PATH, os.path.split(localpath)[-1]), timeout=timeout, verbose=verbose)
        
    def copy_from(self, remotepath, localpath, timeout=10, verbose=False):
        #### The below is commented out since it has a habit of crashing (hence the timeoutCammand wrapper)
        #return timeoutCommand(cmd='python fear_get.py %s %s' % (remotepath, localpath), verbose=verbose).run(timeout=timeout)  
        #### TODO - Make this operating system independent
        #### TODO - Allow port forwarding / tunneling via gate.eng.cam.ac.uk - can we make authentication work?
        if True:#not self.via_gate:
            #### FIXME - This is inefficient - but paramiko's SFTPClient.get operation sometimes hangs bringing down all of python
            if not via_gate:
                cmd = 'scp -i %(rsa_key)s %(username)s@fear:%(remotepath)s %(localpath)s' % {'rsa_key' : LOCAL_TO_REMOTE_KEY_FILE,
                                                                                             'username' : USERNAME,
                                                                                             'remotepath' : remotepath,
                                                                                             'localpath' : localpath}
            else:
                cmd = 'scp -P %(port)d -i %(rsa_key)s %(username)s@localhost:%(remotepath)s %(localpath)s' % {'port' : HOME_TO_REMOTE_PORT, 'rsa_key' : HOME_TO_REMOTE_KEY_FILE,
                                                                                             'username' : USERNAME,
                                                                                             'remotepath' : remotepath,
                                                                                             'localpath' : localpath}                                                                        
            return timeoutCommand(cmd=cmd, verbose=verbose).run(timeout=timeout)
        else:
            #### FIXME - again, this is a hack
            cmd = 'cat %s' % remotepath
            file_contents = self.command(cmd)
            with open(localpath, 'wb') as local_file:
                local_file.writelines(file_contents)
        
    def copy_from_localhost(self, remotepath, localpath, timeout=10, verbose=False):
        #### DO NOT USE!
        #### FIXME - Another hack introduced to deal with gate
        ####       - This is using localhost as a storage machine
        cmd = 'ssh -i %(rsa_key)s %(username)s@%(localhost)s "cat %(remotepath)s ; rm %(remotepath)s"' % {'rsa_key' : GATE_TO_LOCAL_KEY_FILE,
                                                                                                         'username' : USERNAME,
                                                                                                         'localhost' : LOCAL_HOST,
                                                                                                         'remotepath' : remotepath}
        file_contents = self._connection.execute(cmd)
        with open(localpath, 'wb') as local_file:
            local_file.writelines(file_contents)
        
    def rm(self, remote_path):
        #output = self.command('rm %s' % remote_path)
        #return output
        #### FIXME - Correctly handle case of missing file (or anything else that might cause an error)
        try:
            self._connection._sftp.remove(remote_path)
        except:
            return 'Error'    

    def file_exists(self, remote_path):
        #### TODO - Replace this with an ls statement or proper OS independent query
        ####      - Can this be done with SFTP Client instead of SSH - might be faster?
        response = self.command('if [ -e %s ] \nthen \necho ''exists'' \nfi' % remote_path)
        return response == ['exists\n']
    
    def qsub(self, shell_file, verbose=True):
        '''
        Submit a job onto the stack.
        Currently runs jobs from the same folder as they are saved in.
        '''
        fear_string = ' '.join(['. /usr/local/grid/divf2/common/settings.sh;',
                                'cd %s;' % os.path.split(shell_file)[0],
                                'chmod +x %s;' % os.path.split(shell_file)[-1],
                                'qsub -l lr=0',
                                os.path.split(shell_file)[-1] + ';',
                                'cd ..'])
    
        if verbose:
            #print 'Submitting : %s' % fear_string
            print os.path.split(shell_file)[-1]
            
        output_text = self.command(fear_string)
        # Return the job id
        return output_text[0].split(' ')[2]
    
    def qsub_multi(self, shell_files, verbose=True):
        '''
        Submit multiple jobs onto the stack.
        Currently runs jobs from the same folder as they are saved in.
        Warning : assume that all jobs are in the same folder
        '''
        #### TODO - correctly handle case when jobs not in same folder
        fear_string = '. /usr/local/grid/divf2/common/settings.sh; cd %s; ' % os.path.split(shell_files[0])[0]
        fear_string += '; '.join([' '.join(['chmod +x %s;' % os.path.split(shell_file)[-1],
                                            'qsub -l lr=0',
                                            os.path.split(shell_file)[-1]])
                                  for shell_file in shell_files if not shell_file is None])  
        fear_string += '; cd ..'
    
        if verbose:
            print 'Submitting : %s' % fear_string
            
        output_text = self.command(fear_string)
        # Return the job ids
        return [output_text[i].split(' ')[2] for i in range(len(shell_files)) if not shell_files[i] is None]
    
    def qdel(self, job_id):
        output = self.command('. /usr/local/grid/divf2/common/settings.sh; qdel %s' % job_id)
        return output
    
    def qdel_all(self):
        output = self.command('. /usr/local/grid/divf2/common/settings.sh; qdel -u %s' % USERNAME)
        return output
    
    def qstat(self):
        '''Updates a dictionary with (job id, status) pairs'''
        # Parse xml response from qstat
        queue_list = ET.fromstring(''.join(self.command('. /usr/local/grid/divf2/common/settings.sh; qstat -u %s -xml' % USERNAME)))
        # Clear old status dictionary
        self.status = {}
        # Iterate over jobs within multiple queue lists
        for queue in queue_list:
            for job in queue:
                # Iterate over fields in job - extracting id and status
                for element in job:
                    if element.tag == 'JB_job_number':
                        job_id = element.text
                    elif element.tag == 'state':
                        status = element.text
                # Append to dictionary
                self.status[job_id] = status
        
        #### Old non-xml version
        #'''Updates a dictionary with (job id, status) pairs'''
        #output = self.command('. /usr/local/grid/divf2/common/settings.sh; qstat -u %s' % USERNAME)
        ## Now process this text to turn it into a list of job statuses
        ## First remove multiple spaces from the interesting lines (i.e. not header)
        #without_multi_space = [re.sub(' +',' ',line) for line in output[2:]]
        ## Now create a dictionary of job ids and statuses
        #self.status = {key: value for (key, value) in zip([line.split(' ')[0] for line in without_multi_space], \
        #                                                  [line.split(' ')[4] for line in without_multi_space])}

    def qstat_xml(self):
        '''Returns xml output of qstat'''
        return ET.fromstring(''.join(self.command(". /usr/local/grid/divf2/common/settings.sh; qstat -u '*' -xml")))
    
    def job_terminated(self, job_id, update=False):
        '''Returns true if job not listed by qstat'''
        if update:
            self.qstat()
        return not self.status.has_key(job_id)
    
    def job_running(self, job_id, update=False):
        if update:
            self.qstat()
        if self.status.has_key(job_id):
            return self.status[job_id] == 'r'
        else:
            return False
    
    def job_queued(self, job_id, update=False):
        if update:
            self.qstat()
        if self.status.has_key(job_id):
            return self.status[job_id] == 'qw'
        else:
            return False
    
    def job_loading(self, job_id, update=False):
        if update:
            self.qstat()
        if self.status.has_key(job_id):
            return self.status[job_id] == 't'
        else:
            return False
            
    def jobs_running(self, update=True):
        '''Returns number of jobs currently running'''
        if update:
            self.qstat()
        # Count running jobs
        return len([1 for job_id in self.status if self.job_running(job_id)])
            
    def jobs_queued(self, update=True):
        '''Returns number of jobs currently queued'''
        if update:
            self.qstat()
        # Count queued jobs
        return len([1 for job_id in self.status if self.job_queued(job_id)])
            
    def jobs_loading(self, update=True):
        '''Returns number of jobs currently loading'''
        if update:
            self.qstat()
        # Count loading jobs
        return len([1 for job_id in self.status if self.job_loading(job_id)])
        
    def jobs_alive(self, update=True):
        '''Returns number of jobs currently running, queueing or loading'''
        if update:
            self.qstat()
        # Count jobs
        return self.jobs_running(update=False) + self.jobs_queued(update=False) + self.jobs_loading(update=False)
