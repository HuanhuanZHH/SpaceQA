import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image

import dispatch
import roadshow as r


class RoadshowTests(unittest.TestCase):
    def test_type_routing_is_suffix_independent_and_schema_checked(self):
        self.assertEqual(dispatch.identify('AIA_AR_sample_3mogtggl50', {'hek_event_annotation': {}}), 'AIA_AR_sample')
        self.assertEqual(dispatch.identify('EPB_seg_sample_qmn05l7r91', {'science_annotation': {'mask': {}}}), 'EPB_seg_sample')
        self.assertEqual(dispatch.identify('ULF_sample_0yu0mc7d5a', {'ulf_event_annotation': {}}), 'ULF_sample')
        with self.assertRaises(ValueError):
            dispatch.identify('AIA_AR_sample_xxxxxxxxxx', {'ulf_event_annotation': {}})
        with self.assertRaises(ValueError):
            dispatch.identify('P4_arbitrary')

    def test_pairing_rules_do_not_use_clean_or_cross_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'data').mkdir();(root/'label').mkdir()
            (root/'data/a.png').write_bytes(b'x');(root/'data/orphan.png').write_bytes(b'x')
            (root/'label/a.png.json').write_text('{}');(root/'label/missing.png.json').write_text('{}')
            pairs,orphans=r.pair_labels(root,'AIA_AR_sample')
            self.assertEqual(len(pairs),2);self.assertEqual([p.name for p in orphans],['orphan.png'])
            self.assertTrue(any(not data.exists() for _,data in pairs))
            flat=root/'flat';flat.mkdir();(flat/'v.png').write_bytes(b'x');(flat/'v.json').write_text('{}')
            pairs,orphans=r.pair_labels(flat,'EPB_seg_sample')
            self.assertEqual(pairs[0][1].name,'v.png');self.assertEqual(orphans,[])
            ulf=root/'ulf';ulf.mkdir();(ulf/'v.csv').write_text('t,H\n0,1');(ulf/'v.csv.json').write_text('{}')
            pairs,orphans=r.pair_labels(ulf,'ULF_sample')
            self.assertEqual(pairs[0][1].name,'v.csv');self.assertEqual(orphans,[])

    def test_archive_validation_missing_truncated_and_recovered(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source';out=root/'out';source.mkdir();out.mkdir()
            (source/'数据').mkdir();manifest=[]
            for i in range(12):
                name=f'数据/AIA_AR_sample_{i:010d}.tar'
                manifest.append('a'*64+'  '+name)
            (source/'SHA256SUMS.txt').write_text('\n'.join(manifest))
            path=source/'数据/AIA_AR_sample_0000000000.tar';path.write_bytes(b'partial')
            with patch.object(r,'SOURCE',source),patch.object(r,'OUT',out):
                result=r.check_archives()
                self.assertEqual((result['verified'],result['missing'],result['checksum_mismatch']),(0,11,1))
                path.write_bytes(b'complete')
                manifest[0]=dispatch.sha(path)+'  数据/'+path.name
                (source/'SHA256SUMS.txt').write_text('\n'.join(manifest))
                result=r.check_archives()
                self.assertEqual((result['verified'],result['missing'],result['checksum_mismatch']),(1,11,0))

    def test_safe_extract_rejects_traversal_and_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name,kind in [('../outside',tarfile.REGTYPE),('package/link',tarfile.SYMTYPE)]:
                archive=root/'bad.tar'
                with tarfile.open(archive,'w') as stream:
                    info=tarfile.TarInfo(name);info.type=kind;info.linkname='/etc/passwd';stream.addfile(info)
                with self.assertRaises(ValueError):
                    r.safe_extract(archive,root/'dest','package')
            self.assertFalse((root/'outside').exists())

    def test_safe_extract_valid_package_and_read_only_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);archive=root/'good.tar'
            with tarfile.open(archive,'w') as stream:
                info=tarfile.TarInfo('package/label/a.json');value=b'{}';info.size=len(value)
                stream.addfile(info,io.BytesIO(value))
            before=dispatch.sha(archive)
            package=r.safe_extract(archive,root/'extracted','package')
            self.assertEqual((package/'label/a.json').read_text(),'{}')
            self.assertEqual(dispatch.sha(archive),before)

    def test_current_only_preparation_reuses_frozen_features_without_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source';out=root/'output'
            source.mkdir();(source/'数据').mkdir();out.mkdir()
            package='AIA_AR_sample_0000000000';fixture=root/'fixture'/package
            (fixture/'data').mkdir(parents=True);(fixture/'label').mkdir()
            Image.new('RGB',(64,64),(100,80,30)).save(fixture/'data/a.png')
            label=dict(label_header=dict(labelCreateTime='2026-11-01T00:00:00',annotator='test'),
                       basic_annotation=dict(fileName='a.png',startTime='2020-01-01T00:00:00',endTime='2020-01-01T00:00:00'),
                       science_annotation=dict(dateObs='2020-01-01T00:00:00'),
                       hek_event_annotation=dict(matched_events=[],event_count=0,event_types=[],has_events=False))
            (fixture/'label/a.png.json').write_text(json.dumps(label))
            archive=source/'数据'/(package+'.tar')
            with tarfile.open(archive,'w') as stream:
                stream.add(fixture,arcname=package)
            item=dict(package=package,family='AIA_AR_sample',archive=archive.name,expected_sha256=dispatch.sha(archive))
            with patch.object(r,'SOURCE',source),patch.object(r,'OUT',out):
                r.prepare_package(item)
            inventory=list(dispatch.lines(out/'packages'/package/'inventory.jsonl'))
            self.assertEqual(len(inventory),1)
            self.assertEqual(inventory[0]['status'],'prepared',inventory[0].get('error'))
            self.assertEqual(len(inventory[0]['features']),256)
            self.assertFalse(inventory[0]['ground_truth_available'])
            self.assertNotIn('target',inventory[0]);self.assertNotIn('truth_valid',inventory[0])
            self.assertTrue(Path(inventory[0]['image']).exists())
            self.assertEqual(dispatch.sha(archive),item['expected_sha256'])


if __name__=='__main__':
    unittest.main()
